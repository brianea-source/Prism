"""
prism/signal/generator.py
-------------------------
PRISM Signal Generator — combines ML model ensemble with ICC pattern detection.

Pipeline:
  1. Build feature matrix from pipeline.py
  2. Layer 1 ML: direction + confidence (XGBoost/LightGBM classifier)
  3. Layer 2 ML: magnitude estimate (regression)
  4. Layer 3 ML: risk level (RF classifier)
  5. ICC phase detection (M15 bars)
  6. AOI confluence check
  7. SL/TP calculation
  8. R:R validation
  9. Return signal dict or None

Models are loaded from disk (mlflow format).  If models are not yet trained,
the generator falls back to ICC-only signals with reduced confidence.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from prism.signal.icc import AOIDetector, detect_swing_points, get_icc_entry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_RR_RATIO = 1.5          # minimum acceptable risk:reward ratio
MIN_CONFIDENCE = 0.55       # minimum ML confidence to emit a signal
ICC_ONLY_CONFIDENCE = 0.50  # fallback confidence when models not loaded

# Minimum SL distances in pips (per instrument)
MIN_SL_PIPS: dict[str, float] = {
    "EURUSD": 8.0,
    "GBPUSD": 10.0,
    "XAUUSD": 15.0,
    "USDJPY": 10.0,
    "DEFAULT": 8.0,
}

# Pip values
PIP_VALUES: dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "XAUUSD": 0.01,
    "USDJPY": 0.01,
    "DEFAULT": 0.0001,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pip_value(instrument: str) -> float:
    return PIP_VALUES.get(instrument.upper(), PIP_VALUES["DEFAULT"])


def _min_sl_pips(instrument: str) -> float:
    return MIN_SL_PIPS.get(instrument.upper(), MIN_SL_PIPS["DEFAULT"])


class _ModelBundle:
    """Thin wrapper around a set of loaded PRISM ML models."""

    def __init__(self, model_dir: Path, instrument: str) -> None:
        self.instrument = instrument
        self.layer1: Optional[object] = None   # direction classifier
        self.layer2: Optional[object] = None   # magnitude regressor
        self.layer3: Optional[object] = None   # risk classifier
        self._load(model_dir)

    def _load(self, model_dir: Path) -> None:
        """Attempt to load each layer from mlflow artifacts."""
        try:
            import mlflow.sklearn  # type: ignore
        except ImportError:
            logger.warning("mlflow not installed; models will not be loaded")
            return

        for layer_name, attr in [("layer1_direction", "layer1"),
                                   ("layer2_magnitude", "layer2"),
                                   ("layer3_risk", "layer3")]:
            model_path = model_dir / self.instrument / layer_name
            if model_path.exists():
                try:
                    model = mlflow.sklearn.load_model(str(model_path))
                    setattr(self, attr, model)
                    logger.info("Loaded %s from %s", layer_name, model_path)
                except Exception as exc:
                    logger.warning("Could not load %s: %s", model_path, exc)
            else:
                logger.debug("Model not found: %s (will use ICC-only fallback)", model_path)

    @property
    def has_models(self) -> bool:
        return self.layer1 is not None


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    Full PRISM signal generation pipeline.

    Parameters
    ----------
    instrument : str
        PRISM instrument code, e.g. 'EURUSD', 'XAUUSD'.
    model_dir : str
        Path to the model artifact directory (mlflow format).
        Default: 'models/'.
    """

    def __init__(self, instrument: str, model_dir: str = "models/") -> None:
        self.instrument = instrument.upper()
        self.model_dir  = Path(model_dir)
        self._pip = _pip_value(self.instrument)
        self._models = _ModelBundle(self.model_dir, self.instrument)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, df: pd.DataFrame) -> Optional[dict]:
        """
        Run the full signal generation pipeline on the provided OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (columns: datetime/index, open, high, low, close, volume).
            Should contain at least 100 bars for reliable signals.

        Returns
        -------
        dict or None
            Signal dict conforming to the PRISM PRD schema, or None if no
            high-quality signal is found.

            Keys: instrument, signal_time, direction, confidence, entry,
                  sl, tp1, tp2, rr_ratio, risk_level, regime,
                  icc_phase, aoi_confluence, model_version.
        """
        if len(df) < 50:
            logger.warning("Insufficient bars for signal generation (%d < 50)", len(df))
            return None

        # ---- Step 1: Feature pipeline ----------------------------------
        features = self._build_features(df)

        # ---- Step 2-4: ML layers (if models loaded) --------------------
        ml_direction: Optional[int] = None
        ml_confidence: float = 0.0
        ml_magnitude: float  = 0.0
        risk_level: str      = "MEDIUM"
        regime: str          = "UNKNOWN"

        if self._models.has_models and features is not None:
            ml_direction, ml_confidence, ml_magnitude, risk_level, regime = \
                self._run_ml_layers(features)

        # ---- Step 5: ICC detection (M15 chart) -------------------------
        icc_signal = get_icc_entry(df, pip_value=self._pip)
        if icc_signal is None:
            if not self._models.has_models:
                logger.debug("No ICC setup and no ML models; skipping signal")
                return None
            icc_phase   = "NONE"
            icc_entry   = df["close"].iloc[-1]
            icc_sl      = None
        else:
            icc_phase   = icc_signal["phase"]
            icc_entry   = icc_signal["entry"]
            icc_sl      = icc_signal["sl"]

        # Resolve direction from ICC when ML absent / low confidence
        if ml_direction is None:
            if icc_phase == "CONTINUATION_LONG":
                ml_direction  = 1
                ml_confidence = ICC_ONLY_CONFIDENCE
            elif icc_phase == "CONTINUATION_SHORT":
                ml_direction  = -1
                ml_confidence = ICC_ONLY_CONFIDENCE
            else:
                return None

        direction_str = "LONG" if ml_direction == 1 else "SHORT"

        # ---- Step 6: AOI confluence ------------------------------------
        aoi_confluence = False
        try:
            daily_df = self._get_daily_df(df)
            if daily_df is not None and not daily_df.empty:
                aoi = AOIDetector(daily_df)
                aoi_confluence = aoi.is_at_aoi(icc_entry, pip_value=self._pip)
        except Exception as exc:
            logger.warning("AOI check failed: %s", exc)

        # ---- Step 7: SL/TP calculation ---------------------------------
        atr_14 = self._latest_atr(df, period=14)
        sl, tp1, tp2 = self.calculate_sl_tp(
            direction=direction_str,
            entry=icc_entry,
            icc_correction_low=icc_sl if icc_sl is not None else (icc_entry - atr_14 * 1.5),
            atr_14=atr_14,
            instrument=self.instrument,
        )

        # ---- Step 8: R:R validation ------------------------------------
        sl_distance = abs(icc_entry - sl)
        tp1_distance = abs(tp1 - icc_entry)
        rr_ratio = tp1_distance / (sl_distance + 1e-12)

        if rr_ratio < MIN_RR_RATIO:
            logger.info(
                "Signal rejected: R:R %.2f < %.2f minimum (entry=%.5f sl=%.5f tp1=%.5f)",
                rr_ratio, MIN_RR_RATIO, icc_entry, sl, tp1,
            )
            return None

        if ml_confidence < MIN_CONFIDENCE:
            logger.info(
                "Signal rejected: confidence %.3f < %.3f minimum", ml_confidence, MIN_CONFIDENCE
            )
            return None

        # ---- Step 9: Assemble signal -----------------------------------
        signal: dict = {
            "instrument":    self.instrument,
            "signal_time":   datetime.now(tz=timezone.utc).isoformat(),
            "direction":     direction_str,
            "confidence":    round(ml_confidence, 4),
            "entry":         round(icc_entry, 5),
            "sl":            round(sl, 5),
            "tp1":           round(tp1, 5),
            "tp2":           round(tp2, 5),
            "rr_ratio":      round(rr_ratio, 2),
            "risk_level":    risk_level,
            "regime":        regime,
            "icc_phase":     icc_phase,
            "aoi_confluence": aoi_confluence,
            "model_version": self._model_version(),
        }

        logger.info(
            "Signal emitted: %s %s conf=%.2f entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f RR=%.2f AOI=%s",
            signal["instrument"], signal["direction"], signal["confidence"],
            signal["entry"], signal["sl"], signal["tp1"], signal["tp2"],
            signal["rr_ratio"], signal["aoi_confluence"],
        )
        return signal

    def calculate_sl_tp(
        self,
        direction: str,
        entry: float,
        icc_correction_low: float,
        atr_14: float,
        instrument: str,
    ) -> Tuple[float, float, float]:
        """
        Calculate Stop Loss, TP1, and TP2.

        Stop Loss logic:
        - For LONG:  SL = min(icc_correction_low, entry − ATR×1.5), but no
          tighter than ``min_sl_pips`` below entry.
        - For SHORT: mirror of the above.

        TP1 = SL distance × 1.5
        TP2 = SL distance × 3.0

        Parameters
        ----------
        direction : str
            'LONG' or 'SHORT'.
        entry : float
            Entry price.
        icc_correction_low : float
            ICC correction extreme (correction low for LONG, correction high
            for SHORT).
        atr_14 : float
            14-bar ATR in price units.
        instrument : str
            Instrument code (for pip calculation).

        Returns
        -------
        Tuple[float, float, float]
            (sl, tp1, tp2)
        """
        pip = _pip_value(instrument)
        min_sl_distance = _min_sl_pips(instrument) * pip

        if direction.upper() == "LONG":
            icc_sl  = icc_correction_low
            atr_sl  = entry - atr_14 * 1.5
            sl_raw  = max(icc_sl, atr_sl)   # tightest of the two (highest value = least risk)
            sl = min(sl_raw, entry - min_sl_distance)   # enforce minimum distance
            sl_dist = entry - sl
            tp1 = entry + sl_dist * 1.5
            tp2 = entry + sl_dist * 3.0
        else:  # SHORT
            icc_sl  = icc_correction_low   # correction high for shorts
            atr_sl  = entry + atr_14 * 1.5
            sl_raw  = min(icc_sl, atr_sl)  # lowest value = tightest short SL
            sl = max(sl_raw, entry + min_sl_distance)
            sl_dist = sl - entry
            tp1 = entry - sl_dist * 1.5
            tp2 = entry - sl_dist * 3.0

        return round(sl, 5), round(tp1, 5), round(tp2, 5)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Run the feature pipeline on the provided DataFrame."""
        try:
            from prism.data.pipeline import PRISMFeaturePipeline

            # Determine date range from df index or 'datetime' column
            if hasattr(df.index, "date"):
                start = str(df.index[0].date())
                end   = str(df.index[-1].date())
            elif "datetime" in df.columns:
                start = str(pd.to_datetime(df["datetime"].iloc[0]).date())
                end   = str(pd.to_datetime(df["datetime"].iloc[-1]).date())
            else:
                return None

            pipe = PRISMFeaturePipeline(self.instrument)
            features = pipe.build_features(start, end)
            return features

        except Exception as exc:
            logger.warning("Feature pipeline failed in SignalGenerator: %s", exc)
            return None

    def _run_ml_layers(
        self,
        features: pd.DataFrame,
    ) -> Tuple[int, float, float, str, str]:
        """
        Run the three ML layers and return predictions.

        Returns
        -------
        Tuple of (direction, confidence, magnitude, risk_level, regime).
        """
        non_feature = {"datetime", "direction_4h", "magnitude_pips"}
        feat_cols = [c for c in features.columns if c not in non_feature]
        X = features[feat_cols].tail(1).fillna(0).values

        # Layer 1: direction
        direction: int = 0
        confidence: float = 0.0
        try:
            proba = self._models.layer1.predict_proba(X)[0]  # type: ignore[union-attr]
            classes = self._models.layer1.classes_  # type: ignore[union-attr]
            pred_idx = int(np.argmax(proba))
            direction = int(classes[pred_idx])
            confidence = float(proba[pred_idx])
        except Exception as exc:
            logger.warning("Layer 1 prediction failed: %s", exc)

        # Layer 2: magnitude
        magnitude: float = 0.0
        try:
            magnitude = float(self._models.layer2.predict(X)[0])  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("Layer 2 prediction failed: %s", exc)

        # Layer 3: risk level
        risk_level = "MEDIUM"
        try:
            risk_pred = self._models.layer3.predict(X)[0]  # type: ignore[union-attr]
            risk_level = str(risk_pred).upper()
        except Exception as exc:
            logger.debug("Layer 3 prediction failed: %s", exc)

        # Regime: derive from feature values as heuristic
        regime = self._classify_regime(features)

        return direction, confidence, magnitude, risk_level, regime

    def _classify_regime(self, features: pd.DataFrame) -> str:
        """Simple heuristic regime classification from feature values."""
        if "vix_level" not in features.columns:
            return "UNKNOWN"
        vix = features["vix_level"].iloc[-1]
        if pd.isna(vix):
            return "UNKNOWN"
        if vix > 30:
            return "HIGH_VOLATILITY"
        if vix > 20:
            return "ELEVATED"
        return "NORMAL"

    def _latest_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute the most recent ATR value from an OHLCV DataFrame."""
        if len(df) < period + 2:
            return df["high"].iloc[-1] - df["low"].iloc[-1]  # fallback: current bar range
        h = df["high"]
        l = df["low"]
        c = df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def _get_daily_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Resample intraday df to daily OHLCV for AOI detection."""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                return None
            daily = df.resample("D").agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            ).dropna()
            daily.index.name = "date"
            daily.reset_index(inplace=True)
            return daily
        except Exception as exc:
            logger.debug("Daily resample failed: %s", exc)
            return None

    def _model_version(self) -> str:
        """Return a version string for the loaded models."""
        ver_file = self.model_dir / self.instrument / "version.txt"
        if ver_file.exists():
            return ver_file.read_text().strip()
        return "ICC-only-v0"
