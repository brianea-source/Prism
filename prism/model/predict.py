"""
prism/model/predict.py
PRISM Predictor — loads all 4 saved models and returns ensemble signal.
"""
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Module-level path — intentionally a plain variable (not a constant) so tests can
# monkey-patch MODEL_DIR to point at a temporary directory with tiny fixture models.
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

DIRECTION_RMAP = {0: -1, 1: 0, 2: 1}
DIRECTION_STR = {-1: "SHORT", 0: "NEUTRAL", 1: "LONG"}
CONFIDENCE_LEVEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# The four model artefacts PRISMPredictor loads for every instrument.
# Kept as a module-level list so runner startup can verify them without
# having to instantiate a PRISMPredictor (which is expensive and requires
# loading the models into memory).
MODEL_LAYER_NAMES = (
    "layer1_xgb",
    "layer1_lgb",
    "layer2_magnitude",
    "layer3_confidence",
)


def missing_model_files(instruments, model_dir=None) -> list:
    """
    Return the list of ``Path`` objects representing model files that
    DON'T exist for the given instruments. Empty list = every instrument
    has all four layers.

    Used by the runner at startup to refuse to run in CONFIRM / AUTO mode
    when any artefact is missing — otherwise SignalGenerator would raise
    FileNotFoundError mid-scan, losing the signal and spamming Slack with
    stack traces.
    """
    base = Path(model_dir) if model_dir is not None else MODEL_DIR
    missing: list = []
    for inst in instruments:
        for name in MODEL_LAYER_NAMES:
            path = base / f"{name}_{inst}.joblib"
            if not path.exists():
                missing.append(path)
    return missing


class PRISMPredictor:
    """
    Loads all 4 PRISM model artefacts from `models/` and produces per-row
    signal arrays from a feature DataFrame.

    Usage (live — single bar):
        predictor = PRISMPredictor("EURUSD")
        signal = predictor.predict_latest(feature_df)
        # → {"direction": 1, "direction_str": "LONG", "confidence": 0.72,
        #     "confidence_level": "HIGH", "magnitude_pips": 18.4}

    Usage (backtesting — multiple bars):
        result = predictor.predict(feature_df)
        # → {"direction": array([1, -1, 0, ...]), ...}  — one value per input row
    """

    def __init__(self, instrument: str = "EURUSD"):
        self.instrument = instrument
        self._clf_xgb = None
        self._clf_lgb = None
        self._reg = None
        self._clf_rf = None
        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        def _load(name: str):
            # Use the module-level MODEL_DIR so tests can patch it.
            path = MODEL_DIR / f"{name}_{self.instrument}.joblib"
            if not path.exists():
                raise FileNotFoundError(
                    f"Model not found: {path}. Run PRISMTrainer.train_all_layers() first."
                )
            return joblib.load(path)

        self._clf_xgb = _load("layer1_xgb")
        self._clf_lgb = _load("layer1_lgb")
        self._reg = _load("layer2_magnitude")
        self._clf_rf = _load("layer3_confidence")
        logger.info(f"[PRISMPredictor] All 4 models loaded for {self.instrument}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> dict[str, Any]:
        """
        Run all layers on feature row(s).
        For live trading, pass a single-row DataFrame (latest bar).
        For backtesting, pass multiple rows — returns per-row arrays.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns used during training).
            NaN values are filled with 0.

        Returns
        -------
        dict with per-row arrays:
            direction        : np.ndarray  — -1 (short) | 0 (neutral) | 1 (long), one per row
            direction_str    : np.ndarray  — "SHORT" | "NEUTRAL" | "LONG", one per row
            confidence       : np.ndarray  — max ensemble probability, one per row
            confidence_level : np.ndarray  — "LOW" | "MEDIUM" | "HIGH", one per row
            magnitude_pips   : np.ndarray  — expected pips, one per row
        """
        Xf = X.fillna(0).values.astype(np.float32)

        # --- Layer 1: per-row ensemble direction ---
        xgb_proba = self._clf_xgb.predict_proba(Xf)   # shape: (n, 3)
        lgb_proba = self._clf_lgb.predict_proba(Xf)    # shape: (n, 3)
        avg_proba = (xgb_proba + lgb_proba) / 2.0      # shape: (n, 3)

        # Per-row argmax — NOT a mean across rows (which would collapse batch to a single scalar)
        cls_idx = np.argmax(avg_proba, axis=1)          # shape: (n,)
        confidence = avg_proba.max(axis=1)              # shape: (n,) — max proba = confidence

        direction = np.array([DIRECTION_RMAP[c] for c in cls_idx])       # shape: (n,)
        direction_str = np.array([DIRECTION_STR[d] for d in direction])  # shape: (n,)

        # --- Layer 2: per-row magnitude ---
        magnitude_pips = self._reg.predict(Xf)          # shape: (n,)

        # --- Layer 3: per-row confidence tier ---
        conf_tier_raw = self._clf_rf.predict(Xf)        # shape: (n,)
        confidence_level = np.array([CONFIDENCE_LEVEL[max(0, min(2, int(t)))]
                                     for t in conf_tier_raw])            # shape: (n,)

        return {
            "direction": direction,
            "direction_str": direction_str,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "magnitude_pips": magnitude_pips,
        }

    def predict_latest(self, X: pd.DataFrame) -> dict[str, Any]:
        """
        Convenience: predict on the last row of X only.
        Returns scalars rather than arrays — use for live signal generation.

        Example
        -------
            signal = predictor.predict_latest(feature_df)
            if signal["direction"] == 1:
                place_long_order(signal["magnitude_pips"])
        """
        result = self.predict(X.iloc[[-1]])
        return {k: (v[0] if hasattr(v, "__len__") else v) for k, v in result.items()}
