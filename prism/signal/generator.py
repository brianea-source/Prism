"""
PRISM Signal Generator — orchestrates all layers into a complete signal.

Flow:
1. Layer 0: News check (block if high-impact event inside blackout window)
2. Layer 1: H4 regime (XGBoost direction)
3. Layer 2: ICC pattern (H1 structure)
4. Layer 3: FVG zone with M5 break-and-retest confirmation
5. Layer 4: SL/TP calculation
6. Output: SignalPacket ready for execution

This module runs on every new M5/M15 bar.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from prism.signal.fvg import FVGDetector, FVGZone
from prism.signal.icc import ICCDetector
from prism.signal.htf_bias import HTFBiasEngine
from prism.news.intelligence import NewsIntelligence, NewsSignal
from prism.execution.mt5_bridge import SignalPacket

logger = logging.getLogger(__name__)

PIP_SIZE = {"XAUUSD": 0.01, "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01}
MIN_RR = 1.5


class SignalGenerator:
    """Orchestrates all PRISM layers to produce a trade signal."""

    def __init__(self, instrument: str, model_dir: str = "models",
                 persist_fvg: bool = True):
        self.instrument = instrument
        self.model_dir = model_dir
        self.persist_fvg = persist_fvg
        self.news = NewsIntelligence()
        self.icc = ICCDetector()
        self.fvg = FVGDetector(instrument, "H4")
        self.htf_engine = HTFBiasEngine()
        self._predictor = None

    def _load_predictor(self):
        from prism.model.predict import PRISMPredictor
        self._predictor = PRISMPredictor(self.instrument)

    def generate(
        self,
        h4_df: pd.DataFrame,    # H4 OHLCV + features (regime layer)
        h1_df: pd.DataFrame,    # H1 OHLCV (ICC layer)
        entry_df: pd.DataFrame, # M15 or M5 OHLCV (entry layer)
    ) -> Optional[SignalPacket]:
        """
        Run all layers and return SignalPacket if conditions are met.
        Returns None if any layer blocks the trade.
        """
        # --- Layer 0: News Intelligence ---
        news_signal = self.news.get_signal(self.instrument)
        blocked, reason = self.news.should_block_trade(news_signal)
        if blocked:
            logger.info(f"Trade blocked by news: {reason}")
            return None

        # --- Layer 1: H4 Regime (ML Model) ---
        if self._predictor is None:
            self._load_predictor()

        feature_cols = [
            c for c in h4_df.columns
            if c not in ["datetime", "open", "high", "low", "close", "volume",
                         "direction_fwd_4", "magnitude_pips"]
        ]
        if not feature_cols:
            logger.warning("No feature columns found in H4 data")
            return None

        latest_features = h4_df[feature_cols].iloc[[-1]]
        prediction = self._predictor.predict_latest(latest_features)
        direction_int = prediction["direction"]
        confidence = prediction["confidence"]
        direction_str = prediction["direction_str"]

        if direction_int == 0 or confidence < 0.60:
            logger.debug(f"Regime: {direction_str} confidence={confidence:.2f} — no trade")
            return None

        # Align news bias with ML direction
        if news_signal.news_bias != "NEUTRAL":
            if (direction_str == "LONG" and news_signal.news_bias == "BEARISH") or \
               (direction_str == "SHORT" and news_signal.news_bias == "BULLISH"):
                logger.info(
                    f"ML direction ({direction_str}) conflicts with news bias "
                    f"({news_signal.news_bias}) — skipping"
                )
                return None

        # --- HTF Bias Gate (Phase 5) ---
        htf_result = self.htf_engine.refresh(h1_df, h4_df)
        allowed, htf_reason = self.htf_engine.gate_signal(direction_str)
        if not allowed:
            logger.info(f"HTF gate blocked: {htf_reason}")
            return None

        # --- Layer 2: ICC Structure (H1) ---
        icc_signals = self.icc.detect_signals(h1_df)
        active_icc = [s for s in icc_signals if s.get("phase") == "CONTINUATION"]
        if not active_icc:
            logger.debug("No ICC CONTINUATION phase active on H1")
            return None

        icc_signal = active_icc[-1]  # Most recent

        # ICC direction must agree with ML direction
        if icc_signal.get("direction") != direction_str:
            logger.info(
                f"ICC direction ({icc_signal.get('direction')}) disagrees with ML "
                f"({direction_str}) — skipping"
            )
            return None

        # --- Layer 3: FVG Entry (M15/M5) ---
        self.fvg.detect(h4_df)  # Refresh H4 FVG zones
        if self.persist_fvg:
            try:
                self.fvg.save()
            except Exception as e:
                # Persistence must never break signal generation.
                logger.warning(f"FVG save failed: {e}")
        current_price = float(entry_df["close"].iloc[-1])
        fvg_zone = self.fvg.check_entry_trigger(
            current_price, direction_str, m5_df=entry_df
        )

        if fvg_zone is None:
            logger.debug("Price not in active FVG zone with retest — no entry")
            return None

        # --- Layer 4: SL/TP Calculation ---
        entry, sl, tp1, tp2, rr = self._calculate_levels(
            entry_df, direction_str, icc_signal, fvg_zone
        )
        if rr < MIN_RR:
            logger.info(f"RR too low: {rr:.2f} < {MIN_RR} — no trade")
            return None

        logger.info(
            f"SIGNAL: {self.instrument} {direction_str} | entry={entry} sl={sl} "
            f"tp1={tp1} tp2={tp2} rr={rr:.2f} conf={confidence:.2f}"
        )

        # signal_id is auto-assigned via dataclass default_factory (uuid4).
        packet = SignalPacket(
            instrument=self.instrument,
            direction=direction_str,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            rr_ratio=rr,
            confidence=confidence,
            confidence_level=prediction["confidence_level"],
            magnitude_pips=prediction["magnitude_pips"],
            regime=news_signal.risk_regime,
            news_bias=news_signal.news_bias,
            fvg_zone=vars(fvg_zone),
            signal_time=datetime.now(timezone.utc).isoformat(),
            htf_bias={
                "bias_1h": htf_result.bias_1h.value,
                "bias_4h": htf_result.bias_4h.value,
                "aligned": htf_result.aligned,
                "allowed_direction": htf_result.allowed_direction,
                # Last 3 swing types per timeframe — drives the Slack
                # "(HH → HL → HH)" enrichment without leaking the full
                # HTFBiasResult object through the SignalPacket dataclass.
                "swing_seq_1h": [sp["type"] for sp in htf_result.swing_points_1h[-3:]],
                "swing_seq_4h": [sp["type"] for sp in htf_result.swing_points_4h[-3:]],
            },
        )
        return packet

    def _calculate_levels(
        self,
        df: pd.DataFrame,
        direction: str,
        icc_signal: dict,
        fvg_zone: FVGZone,
    ) -> tuple:
        """
        Calculate entry, SL, TP1, TP2, RR.

        SL anchor: the furthest of the ICC correction extreme and the FVG zone
        boundary, plus a small pip buffer. correction_low / correction_high from
        ICCDetector are the *raw* correction extremes (no buffer) — this function
        owns the buffer, so ICC's own stop-loss buffer is not double-counted.
        """
        pip = PIP_SIZE.get(self.instrument, 0.0001)
        latest = df.iloc[-1]
        current_price = float(latest["close"])
        atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else pip * 50

        if direction == "LONG":
            entry = fvg_zone.midline if fvg_zone.midline < current_price else current_price
            icc_low = float(icc_signal.get("correction_low", entry - atr * 1.5))
            sl_anchor = min(icc_low, fvg_zone.bottom)
            sl = sl_anchor - (pip * 5)
            sl_pips = (entry - sl) / pip
            swing_high = float(df["high"].iloc[-20:].max())
            tp1 = max(swing_high, entry + sl_pips * pip * 1.5)
            leg_size = icc_signal.get("leg_size", sl_pips * pip * 2)
            tp2 = entry + leg_size
        else:  # SHORT
            entry = fvg_zone.midline if fvg_zone.midline > current_price else current_price
            icc_high = float(icc_signal.get("correction_high", entry + atr * 1.5))
            sl_anchor = max(icc_high, fvg_zone.top)
            sl = sl_anchor + (pip * 5)
            sl_pips = (sl - entry) / pip
            swing_low = float(df["low"].iloc[-20:].min())
            tp1 = min(swing_low, entry - sl_pips * pip * 1.5)
            leg_size = icc_signal.get("leg_size", sl_pips * pip * 2)
            tp2 = entry - leg_size

        sl_dist = abs(entry - sl)
        tp2_dist = abs(tp2 - entry)
        rr = round(tp2_dist / sl_dist, 2) if sl_dist > 0 else 0.0

        return (
            round(entry, 5), round(sl, 5),
            round(tp1, 5), round(tp2, 5), rr,
        )
