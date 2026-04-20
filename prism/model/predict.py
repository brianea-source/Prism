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

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

DIRECTION_RMAP = {0: -1, 1: 0, 2: 1}
DIRECTION_STR = {-1: "SHORT", 0: "NEUTRAL", 1: "LONG"}
CONFIDENCE_LEVEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


class PRISMPredictor:
    """
    Loads all 4 PRISM model artefacts from `models/` and produces a single
    unified signal dict from a feature DataFrame.

    Usage:
        predictor = PRISMPredictor("EURUSD")
        signal = predictor.predict(X)
        # → {"direction": 1, "direction_str": "LONG", "confidence": 0.72,
        #     "confidence_level": "HIGH", "magnitude_pips": 18.4}
    """

    def __init__(self, instrument: str = "EURUSD"):
        self.instrument = instrument
        self._xgb = None
        self._lgbm = None
        self._reg = None
        self._rf = None
        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        def _load(name: str):
            path = MODELS_DIR / f"{name}_{self.instrument}.joblib"
            if not path.exists():
                raise FileNotFoundError(
                    f"Model not found: {path}. Run PRISMTrainer.train_all_layers() first."
                )
            return joblib.load(path)

        self._xgb = _load("layer1_xgb")
        self._lgbm = _load("layer1_lgbm")
        self._reg = _load("layer2_reg")
        self._rf = _load("layer3_rf")
        logger.info(f"[PRISMPredictor] All 4 models loaded for {self.instrument}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> dict[str, Any]:
        """
        Produce ensemble prediction for the given feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns used during training).
            NaN values are filled with 0.

        Returns
        -------
        dict with keys:
            direction        : int     — -1 (short) | 0 (neutral) | 1 (long)
            direction_str    : str     — "SHORT" | "NEUTRAL" | "LONG"
            confidence       : float   — 0.0 to 1.0
            confidence_level : str     — "LOW" | "MEDIUM" | "HIGH"
            magnitude_pips   : float   — expected move magnitude in pips
        """
        Xf = X.fillna(0).values.astype(np.float32)

        # --- Layer 1: ensemble direction ---
        xgb_proba = self._xgb.predict_proba(Xf)   # (n, 3)
        lgbm_proba = self._lgbm.predict_proba(Xf)  # (n, 3)
        avg_proba = (xgb_proba + lgbm_proba) / 2.0  # average ensemble
        cls_idx = int(np.argmax(avg_proba.mean(axis=0)))  # aggregate across rows
        direction = DIRECTION_RMAP[cls_idx]
        direction_str = DIRECTION_STR[direction]

        # --- Layer 2: magnitude ---
        magnitude_pips = float(np.mean(self._reg.predict(Xf)))

        # --- Layer 3: confidence tier ---
        conf_tier_raw = self._rf.predict(Xf)
        conf_tier = int(round(float(np.mean(conf_tier_raw))))
        conf_tier = max(0, min(2, conf_tier))

        # Map tier → numeric confidence
        tier_to_float = {0: 0.25, 1: 0.55, 2: 0.80}
        confidence = tier_to_float[conf_tier]
        confidence_level = CONFIDENCE_LEVEL[conf_tier]

        return {
            "direction": direction,
            "direction_str": direction_str,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "magnitude_pips": magnitude_pips,
        }

    def predict_row(self, X: pd.DataFrame) -> dict[str, Any]:
        """Convenience: predict on a single row (same signature as predict)."""
        return self.predict(X)
