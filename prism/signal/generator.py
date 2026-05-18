"""
PRISM Signal Generator — orchestrates all layers into a complete signal.

Structure-first flow (PRISM_STRUCTURE_FIRST=1, default):
1. Layer 0: News check (block if high-impact event inside blackout window)
2. Layer 1: HTF bias (1H + 4H swing structure) → determines allowed direction
3. Layer 2: ICC pattern (H1 structure) → finds CONTINUATION setup, confirms direction
4. Layer 3: ML model → provides confidence + magnitude scoring (not direction)
5. Layer 4: FVG zone with M5 break-and-retest confirmation
6. Layer 5: SL/TP calculation
7. Output: SignalPacket ready for execution

Market structure determines direction. ML scores the setup.
News bias adjusts confidence, it does not veto the trade.

This module runs on every new M5/M15 bar.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from prism.signal.fvg import FVGDetector, FVGZone
from prism.signal.icc import ICCDetector
from prism.signal.htf_bias import HTFBiasEngine
from prism.signal.order_blocks import OrderBlockDetector
from prism.signal.po3 import Po3Detector
from prism.signal.sweeps import SweepDetector
from prism.delivery.session_filter import session_label
from prism.news.intelligence import NewsIntelligence, NewsSignal
from prism.execution.mt5_bridge import SignalPacket

logger = logging.getLogger(__name__)

PIP_SIZE = {"XAUUSD": 0.01, "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01}
MIN_RR = 1.5


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip() not in ("", "0", "false", "False")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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
        # Smart money detectors (Phase 6). Always instantiated so observability
        # paths can populate the SignalPacket dict; gating is env-controlled.
        self.ob_detector = OrderBlockDetector(instrument, "H4")
        self.sweep_detector = SweepDetector(instrument)
        self.po3_detector = Po3Detector(instrument)
        # Per-detector failure counters (Phase 6.E). Bumped whenever a smart-money
        # detector raises during ``_evaluate_smart_money``. Surfaced to ops via
        # logger.error + traceback; expose here so a future metrics layer can read
        # ``gen.detector_failure_counts`` without touching the log pipeline. Keys
        # are stable: ``ob`` / ``sweep`` / ``po3``.
        self.detector_failure_counts: dict[str, int] = {
            "ob": 0, "sweep": 0, "po3": 0,
        }
        self._predictor = None
        self.last_rejection_gate: Optional[str] = None

    def _load_predictor(self):
        from prism.model.predict import PRISMPredictor
        self._predictor = PRISMPredictor(self.instrument)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        h4_df: pd.DataFrame,    # H4 OHLCV + features (regime layer)
        h1_df: pd.DataFrame,    # H1 OHLCV (ICC layer)
        entry_df: pd.DataFrame, # M15 or M5 OHLCV (entry layer)
    ) -> Optional[SignalPacket]:
        """Run all layers and return SignalPacket if conditions are met.

        Dispatches to the structure-first flow (default) or legacy
        ML-direction flow based on ``PRISM_STRUCTURE_FIRST`` env var.
        """
        if _env_bool("PRISM_STRUCTURE_FIRST", "1"):
            return self._generate_structure_first(h4_df, h1_df, entry_df)
        return self._generate_legacy(h4_df, h1_df, entry_df)

    # ------------------------------------------------------------------
    # Structure-first flow (new default)
    # ------------------------------------------------------------------

    def _generate_structure_first(
        self,
        h4_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        entry_df: pd.DataFrame,
    ) -> Optional[SignalPacket]:
        """Market structure determines direction. ML scores the setup.

        Flow:
        1. News blackout check
        2. HTF bias (1H + 4H swings) → allowed direction
        3. ICC CONTINUATION on H1 → trade direction (must agree with HTF)
        4. ML model → confidence + magnitude (not direction)
        5. News bias → adjusts confidence (soft penalty, not a veto)
        6. FVG entry trigger
        7. Smart money confluence
        8. SL/TP + RR check
        """
        self.last_rejection_gate = None

        # --- Layer 0: News blackout ---
        news_signal = self.news.get_signal(self.instrument)
        blocked, reason = self.news.should_block_trade(news_signal)
        if blocked:
            self.last_rejection_gate = "news_blackout"
            logger.info("Trade blocked by news: %s", reason)
            return None

        # --- Layer 1: HTF bias → direction lock ---
        htf_result = self.htf_engine.refresh(h1_df, h4_df)
        if not htf_result.aligned or htf_result.allowed_direction is None:
            self.last_rejection_gate = "htf_bias"
            logger.info(
                "HTF not aligned: 1H=%s 4H=%s — no direction lock",
                htf_result.bias_1h.value, htf_result.bias_4h.value,
            )
            return None
        direction_str = htf_result.allowed_direction  # "LONG" or "SHORT"

        # --- Layer 2: ICC structure (H1) ---
        icc_signals = self.icc.detect_signals(h1_df)
        active_icc = [s for s in icc_signals if s.get("phase") == "CONTINUATION"]
        if not active_icc:
            self.last_rejection_gate = "icc_structure"
            logger.debug("No ICC CONTINUATION phase active on H1")
            return None

        icc_signal = active_icc[-1]

        if icc_signal.get("direction") != direction_str:
            self.last_rejection_gate = "icc_direction"
            logger.info(
                "ICC direction (%s) disagrees with HTF bias (%s) — skipping",
                icc_signal.get("direction"), direction_str,
            )
            return None

        # --- Layer 3: ML confidence + magnitude ---
        if self._predictor is None:
            self._load_predictor()

        feature_cols = [
            c for c in h4_df.columns
            if c not in ["datetime", "open", "high", "low", "close", "volume",
                         "direction_fwd_4", "magnitude_pips"]
        ]
        if not feature_cols:
            self.last_rejection_gate = "no_features"
            logger.warning("No feature columns found in H4 data")
            return None

        latest_features = h4_df[feature_cols].iloc[[-1]]
        prediction = self._predictor.predict_latest(latest_features)
        confidence = float(prediction["confidence"])
        ml_direction = prediction["direction_str"]

        min_conf = _env_float("PRISM_MIN_CONFIDENCE", 0.60)
        if confidence < min_conf:
            self.last_rejection_gate = "ml_confidence"
            logger.debug(
                "ML confidence %.2f below threshold %.2f — no trade",
                confidence, min_conf,
            )
            return None

        # --- News bias: soft confidence adjustment, not a veto ---
        news_penalty = _env_float("PRISM_NEWS_BIAS_PENALTY", 0.10)
        if news_signal.news_bias != "NEUTRAL":
            bias_agrees = (
                (direction_str == "LONG" and news_signal.news_bias == "BULLISH") or
                (direction_str == "SHORT" and news_signal.news_bias == "BEARISH")
            )
            if not bias_agrees:
                confidence = max(0.0, confidence - news_penalty)
                logger.info(
                    "News bias (%s) opposes structure direction (%s) — "
                    "confidence reduced by %.0f%% to %.2f",
                    news_signal.news_bias, direction_str,
                    news_penalty * 100, confidence,
                )
                if confidence < min_conf:
                    self.last_rejection_gate = "news_bias_penalty"
                    logger.info(
                        "Confidence %.2f dropped below threshold after "
                        "news penalty — skipping",
                        confidence,
                    )
                    return None

        # If ML agrees on direction, log it; if not, the trade still proceeds
        # because structure (HTF + ICC) is the authority on direction.
        if ml_direction != direction_str:
            logger.info(
                "ML predicted %s but structure says %s — trading with "
                "structure (confidence=%.2f)",
                ml_direction, direction_str, confidence,
            )

        # --- Layer 4: FVG Entry (M15/M5) ---
        self.fvg.detect(h4_df)
        if self.persist_fvg:
            try:
                self.fvg.save()
            except Exception as e:
                logger.warning("FVG save failed: %s", e)
        current_price = float(entry_df["close"].iloc[-1])
        fvg_zone = self.fvg.check_entry_trigger(
            current_price, direction_str, m5_df=entry_df
        )

        if fvg_zone is None:
            self.last_rejection_gate = "fvg_entry"
            logger.debug("Price not in active FVG zone with retest — no entry")
            return None

        # --- Smart Money Layer (Phase 6.D) ---
        smart_money = self._evaluate_smart_money(
            h4_df=h4_df,
            entry_df=entry_df,
            current_price=current_price,
            direction_str=direction_str,
        )
        if smart_money is not None and smart_money.get("blocked"):
            self.last_rejection_gate = "smart_money"
            logger.info(
                "Smart-money gate blocked: %s", smart_money.get("block_reason"),
            )
            return None

        # --- Layer 5: SL/TP Calculation ---
        entry, sl, tp1, tp2, rr = self._calculate_levels(
            entry_df, direction_str, icc_signal, fvg_zone
        )
        if rr < MIN_RR:
            self.last_rejection_gate = "rr_ratio"
            logger.info("RR too low: %.2f < %.1f — no trade", rr, MIN_RR)
            return None

        logger.info(
            "SIGNAL: %s %s | entry=%s sl=%s tp1=%s tp2=%s rr=%.2f conf=%.2f "
            "ml_dir=%s",
            self.instrument, direction_str, entry, sl, tp1, tp2, rr,
            confidence, ml_direction,
        )

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
                "swing_seq_1h": [sp["type"] for sp in htf_result.swing_points_1h[-3:]],
                "swing_seq_4h": [sp["type"] for sp in htf_result.swing_points_4h[-3:]],
            },
            smart_money=(
                {k: v for k, v in smart_money.items() if k not in ("blocked", "block_reason")}
                if smart_money is not None
                else None
            ),
        )
        return packet

    # ------------------------------------------------------------------
    # Legacy ML-direction flow (PRISM_STRUCTURE_FIRST=0)
    # ------------------------------------------------------------------

    def _generate_legacy(
        self,
        h4_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        entry_df: pd.DataFrame,
    ) -> Optional[SignalPacket]:
        """Original flow: ML model predicts direction, everything else confirms.

        Preserved behind PRISM_STRUCTURE_FIRST=0 for rollback safety.
        """
        self.last_rejection_gate = None

        # --- Layer 0: News Intelligence ---
        news_signal = self.news.get_signal(self.instrument)
        blocked, reason = self.news.should_block_trade(news_signal)
        if blocked:
            self.last_rejection_gate = "news_blackout"
            logger.info("Trade blocked by news: %s", reason)
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
            self.last_rejection_gate = "no_features"
            logger.warning("No feature columns found in H4 data")
            return None

        latest_features = h4_df[feature_cols].iloc[[-1]]
        prediction = self._predictor.predict_latest(latest_features)
        direction_int = prediction["direction"]
        confidence = prediction["confidence"]
        direction_str = prediction["direction_str"]

        legacy_min_conf = _env_float("PRISM_MIN_CONFIDENCE", 0.60)
        if direction_int == 0 or confidence < legacy_min_conf:
            self.last_rejection_gate = "ml_confidence"
            logger.debug("Regime: %s confidence=%.2f — no trade", direction_str, confidence)
            return None

        # Align news bias with ML direction
        if news_signal.news_bias != "NEUTRAL":
            if (direction_str == "LONG" and news_signal.news_bias == "BEARISH") or \
               (direction_str == "SHORT" and news_signal.news_bias == "BULLISH"):
                self.last_rejection_gate = "news_bias"
                logger.info(
                    "ML direction (%s) conflicts with news bias (%s) — skipping",
                    direction_str, news_signal.news_bias,
                )
                return None

        # --- HTF Bias Gate (Phase 5) ---
        htf_result = self.htf_engine.refresh(h1_df, h4_df)
        allowed, htf_reason = self.htf_engine.gate_signal(direction_str)
        if not allowed:
            self.last_rejection_gate = "htf_bias"
            logger.info("HTF gate blocked: %s", htf_reason)
            return None

        # --- Layer 2: ICC Structure (H1) ---
        icc_signals = self.icc.detect_signals(h1_df)
        active_icc = [s for s in icc_signals if s.get("phase") == "CONTINUATION"]
        if not active_icc:
            self.last_rejection_gate = "icc_structure"
            logger.debug("No ICC CONTINUATION phase active on H1")
            return None

        icc_signal = active_icc[-1]

        if icc_signal.get("direction") != direction_str:
            self.last_rejection_gate = "icc_direction"
            logger.info(
                "ICC direction (%s) disagrees with ML (%s) — skipping",
                icc_signal.get("direction"), direction_str,
            )
            return None

        # --- Layer 3: FVG Entry (M15/M5) ---
        self.fvg.detect(h4_df)
        if self.persist_fvg:
            try:
                self.fvg.save()
            except Exception as e:
                logger.warning("FVG save failed: %s", e)
        current_price = float(entry_df["close"].iloc[-1])
        fvg_zone = self.fvg.check_entry_trigger(
            current_price, direction_str, m5_df=entry_df
        )

        if fvg_zone is None:
            self.last_rejection_gate = "fvg_entry"
            logger.debug("Price not in active FVG zone with retest — no entry")
            return None

        # --- Smart Money Layer (Phase 6.D) ---
        smart_money = self._evaluate_smart_money(
            h4_df=h4_df,
            entry_df=entry_df,
            current_price=current_price,
            direction_str=direction_str,
        )
        if smart_money is not None and smart_money.get("blocked"):
            self.last_rejection_gate = "smart_money"
            logger.info(
                "Smart-money gate blocked: %s", smart_money.get("block_reason"),
            )
            return None

        # --- Layer 4: SL/TP Calculation ---
        entry, sl, tp1, tp2, rr = self._calculate_levels(
            entry_df, direction_str, icc_signal, fvg_zone
        )
        if rr < MIN_RR:
            self.last_rejection_gate = "rr_ratio"
            logger.info("RR too low: %.2f < %.1f — no trade", rr, MIN_RR)
            return None

        logger.info(
            "SIGNAL: %s %s | entry=%s sl=%s tp1=%s tp2=%s rr=%.2f conf=%.2f",
            self.instrument, direction_str, entry, sl, tp1, tp2, rr, confidence,
        )

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
                "swing_seq_1h": [sp["type"] for sp in htf_result.swing_points_1h[-3:]],
                "swing_seq_4h": [sp["type"] for sp in htf_result.swing_points_4h[-3:]],
            },
            smart_money=(
                {k: v for k, v in smart_money.items() if k not in ("blocked", "block_reason")}
                if smart_money is not None
                else None
            ),
        )
        return packet

    # ------------------------------------------------------------------
    # Phase 6.D: smart-money confluence
    # ------------------------------------------------------------------
    def _evaluate_smart_money(
        self,
        h4_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        current_price: float,
        direction_str: str,
    ) -> Optional[dict]:
        """Run OB / Sweep / Po3 detectors and return a dict for ``SignalPacket.smart_money``.

        Returns ``None`` when the master switch ``PRISM_SMART_MONEY_ENABLED`` is
        off, so existing flows are unaffected. When enabled, the dict always has
        ``ob``/``sweep``/``po3`` sub-dicts (or ``None`` for each missing piece) plus
        ``blocked`` + ``block_reason`` for the caller to act on.
        """
        if not _env_bool("PRISM_SMART_MONEY_ENABLED", "0"):
            return None

        sweep_required = _env_bool("PRISM_SWEEP_REQUIRED", "1")
        po3_required = _env_bool("PRISM_PO3_REQUIRED", "1")
        min_disp_pips = _env_float("PRISM_MIN_DISPLACEMENT_PIPS", 10.0)
        ob_max_dist_pips = _env_float("PRISM_OB_MAX_DISTANCE_PIPS", 30.0)

        # --- Order Blocks (H4) -----------------------------------------
        # Detector exceptions must never crash signal generation, but they MUST
        # be loud — a silently-broken detector during observability rollout is
        # exactly the failure mode this layer is meant to prevent. Log at error
        # level with the full traceback (exc_info=True) and bump a counter ops
        # can scrape via ``gen.detector_failure_counts``.
        try:
            self.ob_detector.detect(h4_df, min_displacement_pips=min_disp_pips)
            self.ob_detector.update_states(h4_df)
            nearest = self.ob_detector.get_nearest_ob(current_price, direction_str)
            ob_distance = self.ob_detector.distance_to_ob(current_price, direction_str)
        except Exception as exc:
            self.detector_failure_counts["ob"] += 1
            logger.error(
                f"OrderBlockDetector failed for {self.instrument}: {exc}",
                exc_info=True,
            )
            nearest = None
            ob_distance = None

        ob_dict: Optional[dict] = None
        if nearest is not None:
            ob_dict = {
                "state": nearest.state.value,
                "direction": nearest.direction,
                "effective_direction": nearest.effective_direction,
                "high": nearest.high,
                "low": nearest.low,
                "midpoint": nearest.midpoint,
                "timeframe": nearest.timeframe,
                "distance_pips": ob_distance,
                "is_rejection_block": nearest.is_rejection_block,
                "in_range": (
                    ob_distance is not None and ob_distance <= ob_max_dist_pips
                ),
            }

        # --- Liquidity Sweep (entry timeframe) -------------------------
        # On exception we set ``has_recent=False`` deliberately: a broken sweep
        # detector with ``PRISM_SWEEP_REQUIRED=1`` is treated as "no qualifying
        # sweep" and the gate blocks. This is fail-closed by design — better
        # to skip a trade than to fire one without confluence we can't compute.
        try:
            self.sweep_detector.detect(entry_df)
            last_sweep = self.sweep_detector.last_sweep(direction_str)
            has_recent = self.sweep_detector.has_recent_sweep(
                direction_str, bars_back=5, require_displacement=True
            )
        except Exception as exc:
            self.detector_failure_counts["sweep"] += 1
            logger.error(
                f"SweepDetector failed for {self.instrument}: {exc}",
                exc_info=True,
            )
            last_sweep = None
            has_recent = False

        sweep_dict: Optional[dict] = None
        if last_sweep is not None:
            anchor = (
                self.sweep_detector._latest_scanned_bar
                if self.sweep_detector._latest_scanned_bar is not None
                else last_sweep.sweep_bar
            )
            sweep_dict = {
                "type": last_sweep.type,
                "swept_level": last_sweep.swept_level,
                "sweep_bar": last_sweep.sweep_bar,
                "bars_ago": int(anchor - last_sweep.sweep_bar),
                "displacement_followed": last_sweep.displacement_followed,
                "timestamp": last_sweep.timestamp,
                "qualifies": has_recent,
            }

        # --- Po3 phase (entry timeframe, current session) --------------
        # Same fail-closed contract as the sweep gate.
        session_str = session_label(datetime.now(timezone.utc))
        try:
            po3_state = self.po3_detector.detect_phase(entry_df, session=session_str)
            is_entry_phase = self.po3_detector.is_entry_phase(po3_state)
        except Exception as exc:
            self.detector_failure_counts["po3"] += 1
            logger.error(
                f"Po3Detector failed for {self.instrument}: {exc}",
                exc_info=True,
            )
            po3_state = None
            is_entry_phase = False

        po3_dict: Optional[dict] = None
        if po3_state is not None:
            po3_dict = {
                "phase": po3_state.phase.value,
                "session": po3_state.session,
                "range_size_pips": po3_state.range_size_pips,
                "sweep_detected": po3_state.sweep_detected,
                "displacement_detected": po3_state.displacement_detected,
                "is_entry_phase": is_entry_phase,
            }

        # --- Gating ----------------------------------------------------
        # Fail-fast ``elif`` chain by design: any single gate failure blocks
        # the signal, and ``block_reason`` reports the first one in ICT order
        # of operations — sweep (manipulation) must complete before Po3
        # (distribution), so a missing sweep is a strictly upstream failure
        # and that's the more actionable diagnostic to surface. Both gates
        # still BOTH have to pass for an entry to fire; the elif only changes
        # which reason wins the log line.
        blocked = False
        block_reason = ""
        if sweep_required and not has_recent:
            blocked = True
            block_reason = "no recent qualifying sweep for direction"
        elif po3_required and not is_entry_phase:
            blocked = True
            phase_val = po3_dict["phase"] if po3_dict else "UNKNOWN"
            block_reason = f"Po3 phase {phase_val} is not entry phase"

        return {
            "ob": ob_dict,
            "sweep": sweep_dict,
            "po3": po3_dict,
            "blocked": blocked,
            "block_reason": block_reason,
        }

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
