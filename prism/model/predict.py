"""
prism/model/predict.py
PRISM Predictor — loads all 4 saved models and returns ensemble signal.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Module-level path — intentionally a plain variable (not a constant) so tests can
# monkey-patch MODEL_DIR to point at a temporary directory with tiny fixture models.
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

#: Filename suffix for the per-instrument JSON sidecar that locks
#: training-time configuration into the model artifact. Lives next to
#: the joblibs at ``models/manifest_<instrument>.json``. Used to detect
#: ``PRISM_OB_MAX_DISTANCE_PIPS`` drift between train and live —
#: PHASE_7A_SCOPE.md §2.4 / §8.1.
MANIFEST_FILENAME_TEMPLATE = "manifest_{instrument}.json"

#: Filename template for the per-instrument feature-columns sidecar
#: written by :func:`prism.model.train.write_feature_cols`. Locks the
#: ordered list of training-time feature columns so predict-time can
#: project the live feature frame onto the trained schema. Required
#: for every model on disk after this PR — legacy joblibs without a
#: sidecar fail loud at predictor load with a retrain instruction.
FEATURE_COLS_FILENAME_TEMPLATE = "feature_cols_{instrument}.json"

#: When ``True`` (env: ``PRISM_OB_MAX_DISTANCE_PIPS_STRICT=1``), a
#: train/live mismatch on ``PRISM_OB_MAX_DISTANCE_PIPS`` raises at
#: model load. Default ``False`` for the first deploy cycle so the
#: warning is non-fatal — flip after one retrain has written a
#: manifest, per scope §2.4.
def _strict_ob_distance_pips() -> bool:
    return os.environ.get("PRISM_OB_MAX_DISTANCE_PIPS_STRICT", "0") == "1"

DIRECTION_RMAP = {0: -1, 1: 0, 2: 1}
DIRECTION_STR = {-1: "SHORT", 0: "NEUTRAL", 1: "LONG"}
CONFIDENCE_LEVEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# The four model artefacts PRISMPredictor loads for every instrument.
# Kept as a module-level list so runner startup can verify them without
# having to instantiate a PRISMPredictor (which is expensive and requires
# loading the models into memory).
MODEL_LAYER_NAMES = (
    "layer1_xgb",
    "layer1_lgbm",
    "layer2_reg",
    "layer3_rf",
)


def manifest_path(instrument: str, model_dir: Optional[Path] = None) -> Path:
    """Resolve the ``models/manifest_<instrument>.json`` path."""
    base = Path(model_dir) if model_dir is not None else MODEL_DIR
    return base / MANIFEST_FILENAME_TEMPLATE.format(instrument=instrument)


def read_manifest(
    instrument: str, model_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Read the per-instrument model manifest.

    Returns the parsed dict on success. ``None`` when the manifest
    doesn't exist (legacy models pre-Phase-7A) or when JSON parsing
    fails — the caller is responsible for deciding whether absence is
    fatal.
    """
    path = manifest_path(instrument, model_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not read manifest %s: %s — treating as missing", path, exc,
        )
        return None


def write_manifest(
    instrument: str,
    *,
    ob_max_distance_pips: float,
    phase7a_features_active: bool = False,
    extra: Optional[dict] = None,
    model_dir: Optional[Path] = None,
) -> Path:
    """Write a per-instrument manifest sidecar at training time.

    Locks the env-derived config that the model was trained against
    so :func:`validate_manifest_against_env` can detect drift at
    inference time. Idempotent — overwrites if present.
    """
    path = manifest_path(instrument, model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instrument": instrument,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "ob_max_distance_pips": float(ob_max_distance_pips),
        "phase7a_features_active": bool(phase7a_features_active),
    }
    if extra:
        # extra never overrides the canonical keys above
        for k, v in extra.items():
            if k not in payload:
                payload[k] = v
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote model manifest → %s", path)
    return path


def validate_manifest_against_env(
    manifest: Optional[dict], *, instrument: str,
) -> Optional[str]:
    """Compare the manifest's locked config to the live environment.

    Returns ``None`` when no drift is detected. Returns a human-readable
    message describing the drift otherwise. Caller decides whether to
    log a warning or raise based on
    :data:`PRISM_OB_MAX_DISTANCE_PIPS_STRICT`. Pre-Phase-7A models
    that lack a manifest are silently allowed (no message) — this keeps
    the rollout backward-compatible while operators retrain.
    """
    if manifest is None:
        return None

    locked_raw = manifest.get("ob_max_distance_pips")
    if locked_raw is None:
        return None
    try:
        locked = float(locked_raw)
    except (TypeError, ValueError):
        return (
            f"manifest for {instrument} has malformed ob_max_distance_pips: "
            f"{locked_raw!r}"
        )

    live_raw = os.environ.get("PRISM_OB_MAX_DISTANCE_PIPS", "30.0")
    try:
        live = float(live_raw)
    except ValueError:
        return (
            f"PRISM_OB_MAX_DISTANCE_PIPS={live_raw!r} could not be parsed as float; "
            f"manifest has {locked}"
        )

    # Float equality with a tolerance — env vars round-trip via str so
    # exact equality is fine for typical configs (30, 30.0, 25.5), but
    # a tiny tolerance shields against accumulator drift on exotic
    # values an operator might paste with extra precision.
    if abs(live - locked) > 1e-6:
        return (
            f"PRISM_OB_MAX_DISTANCE_PIPS drift for {instrument}: "
            f"manifest locked {locked}, runtime env is {live}. "
            "Retrain with the current env, or revert the env to the "
            "trained value. PHASE_7A_SCOPE.md §2.4."
        )
    return None


def feature_cols_path(instrument: str, model_dir: Optional[Path] = None) -> Path:
    """Resolve the ``models/feature_cols_<instrument>.json`` path."""
    base = Path(model_dir) if model_dir is not None else MODEL_DIR
    return base / FEATURE_COLS_FILENAME_TEMPLATE.format(instrument=instrument)


def read_feature_cols(
    instrument: str, model_dir: Optional[Path] = None,
) -> Optional[list]:
    """Read the ordered list of training-time feature column names.

    Returns the parsed list on success. ``None`` when the sidecar
    doesn't exist (legacy models pre-feature-alignment) or when JSON
    parsing fails — callers decide whether absence is fatal.
    """
    path = feature_cols_path(instrument, model_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not read feature_cols sidecar %s: %s — treating as missing",
            path, exc,
        )
        return None
    cols = payload.get("feature_cols")
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        logger.warning(
            "feature_cols sidecar %s has malformed schema (expected list[str]); "
            "got %r",
            path, type(cols).__name__,
        )
        return None
    return cols


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
        # The feature_cols sidecar is also required — without it the
        # predictor refuses to load (fail loud) because we can't
        # safely project the live feature frame onto an unknown schema.
        fc_path = feature_cols_path(inst, model_dir=base)
        if not fc_path.exists():
            missing.append(fc_path)
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
        self._feature_cols: list[str] | None = None
        # Track whether we've already logged drift this session so the
        # WARN line appears once per predictor, not once per bar.
        self._drift_warned: bool = False
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
        self._clf_lgb = _load("layer1_lgbm")
        self._reg = _load("layer2_reg")
        self._clf_rf = _load("layer3_rf")

        # Load the feature-columns sidecar. Required — without it we
        # can't safely project the live feature frame onto the trained
        # schema, and a silent column-count drift in production would
        # crash mid-scan with the cryptic XGBoost message
        # ``Feature shape mismatch, expected: N, got M``. Fail loud
        # at load time with a clear retrain instruction instead.
        feature_cols = read_feature_cols(self.instrument, model_dir=MODEL_DIR)
        if feature_cols is None:
            raise FileNotFoundError(
                f"feature_cols sidecar missing for {self.instrument}: expected "
                f"{feature_cols_path(self.instrument)}. Retrain this instrument "
                f"with `python -m prism.model.retrain --instrument {self.instrument}` "
                "to generate it. Legacy joblibs predating the feature-alignment "
                "fix are not supported — they would silently mis-predict."
            )
        self._feature_cols = feature_cols
        logger.info(
            f"[PRISMPredictor] All 4 models + {len(feature_cols)}-col feature schema "
            f"loaded for {self.instrument}"
        )

        # Phase 7.A sidecar lock-in: detect train/live env drift on
        # ``PRISM_OB_MAX_DISTANCE_PIPS``. Warn-only by default — flip
        # ``PRISM_OB_MAX_DISTANCE_PIPS_STRICT=1`` after the first
        # retrain has written a manifest. PHASE_7A_SCOPE.md §2.4.
        manifest = read_manifest(self.instrument)
        self._manifest = manifest
        drift_message = validate_manifest_against_env(
            manifest, instrument=self.instrument,
        )
        if drift_message:
            if _strict_ob_distance_pips():
                raise RuntimeError(
                    f"[PRISMPredictor] {drift_message} "
                    "(PRISM_OB_MAX_DISTANCE_PIPS_STRICT=1)"
                )
            logger.warning("[PRISMPredictor] %s", drift_message)

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
        X = self._project_to_trained_schema(X)
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

    # ------------------------------------------------------------------
    # Feature-schema projection
    # ------------------------------------------------------------------

    def _project_to_trained_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder + restrict ``X`` to match the training-time feature
        schema. Drops any columns not seen during training and
        zero-fills any expected columns that are missing from the live
        frame. Logs a single WARNING per predictor session listing the
        drift, so operators can see what is happening in production
        without the runner crash-looping on shape mismatches.

        This is the safety net for the entire upstream feature
        pipeline: even if the train and live ``_engineer`` paths fall
        out of sync (FRED outage, Phase 7.A sidecar toggle, an extra
        cache column like ``source`` leaking through, etc.), the
        predictor will still run on the trained-column subset and
        surface the drift in the log instead of crashing mid-scan.
        """
        if self._feature_cols is None:
            # Should be unreachable — _load_models raises when the
            # sidecar is missing — but guard anyway so this method is
            # safe to call standalone.
            return X

        trained = self._feature_cols
        live_cols = list(X.columns)

        if not self._drift_warned:
            trained_set = set(trained)
            live_set = set(live_cols)
            missing = [c for c in trained if c not in live_set]
            unexpected = [c for c in live_cols if c not in trained_set]
            if missing or unexpected:
                logger.warning(
                    "[PRISMPredictor:%s] feature drift detected at predict time: "
                    "expected %d columns, live frame has %d. "
                    "missing_in_live=%s extra_in_live=%s. "
                    "Projecting to trained schema (zero-fill missing, drop extra). "
                    "Retrain to eliminate the drift.",
                    self.instrument, len(trained), len(live_cols),
                    missing or "[]", unexpected or "[]",
                )
                self._drift_warned = True

        # ``reindex`` both reorders to the trained order AND drops
        # extras AND zero-fills missing columns in a single pass.
        return X.reindex(columns=trained, fill_value=0)
