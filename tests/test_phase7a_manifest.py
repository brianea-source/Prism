"""Phase 7.A manifest lock-in tests.

Covers the train/live drift detection on ``PRISM_OB_MAX_DISTANCE_PIPS``
that ``retrain.py`` writes into the model artifact sidecar and
``predict.py`` validates at load time. PHASE_7A_SCOPE.md §2.4 / §8.1.

This test file does NOT exercise the joblib loading path (that lives
in ``test_phase4_live_bars.py`` already) — it isolates the manifest
helpers and the strict-mode env switch so the lock-in surface is
covered without dragging in the full predictor.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism.model.predict import (
    MANIFEST_FILENAME_TEMPLATE,
    manifest_path,
    read_manifest,
    validate_manifest_against_env,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestManifestPath:

    def test_default_dir(self):
        path = manifest_path("EURUSD")
        assert path.name == MANIFEST_FILENAME_TEMPLATE.format(instrument="EURUSD")

    def test_custom_dir(self, tmp_path):
        path = manifest_path("EURUSD", model_dir=tmp_path)
        assert path == tmp_path / "manifest_EURUSD.json"

    def test_per_instrument_isolation(self, tmp_path):
        a = manifest_path("EURUSD", model_dir=tmp_path)
        b = manifest_path("GBPUSD", model_dir=tmp_path)
        assert a != b


# ---------------------------------------------------------------------------
# Read / write round-trip
# ---------------------------------------------------------------------------


class TestManifestIO:

    def test_write_creates_file_with_canonical_keys(self, tmp_path):
        write_manifest(
            "EURUSD",
            ob_max_distance_pips=30.0,
            phase7a_features_active=True,
            model_dir=tmp_path,
        )
        path = tmp_path / "manifest_EURUSD.json"
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["instrument"] == "EURUSD"
        assert payload["ob_max_distance_pips"] == 30.0
        assert payload["phase7a_features_active"] is True
        assert "trained_at" in payload

    def test_write_creates_parent_directory(self, tmp_path):
        deep = tmp_path / "deeply" / "nested" / "models"
        write_manifest(
            "EURUSD", ob_max_distance_pips=30.0, model_dir=deep,
        )
        assert (deep / "manifest_EURUSD.json").exists()

    def test_write_overwrites_existing(self, tmp_path):
        write_manifest(
            "EURUSD", ob_max_distance_pips=30.0, model_dir=tmp_path,
        )
        write_manifest(
            "EURUSD", ob_max_distance_pips=42.0, model_dir=tmp_path,
        )
        m = read_manifest("EURUSD", model_dir=tmp_path)
        assert m["ob_max_distance_pips"] == 42.0

    def test_read_returns_none_for_missing(self, tmp_path):
        assert read_manifest("EURUSD", model_dir=tmp_path) is None

    def test_read_returns_none_for_malformed_json(self, tmp_path):
        path = tmp_path / "manifest_EURUSD.json"
        path.write_text("not valid json {{{")
        assert read_manifest("EURUSD", model_dir=tmp_path) is None

    def test_extra_fields_persisted_but_not_overriding_canonical(self, tmp_path):
        write_manifest(
            "EURUSD",
            ob_max_distance_pips=30.0,
            extra={
                "retrain_report": "/tmp/report.json",
                "instrument": "WRONG",  # canonical wins
                "ob_max_distance_pips": 999,  # canonical wins
            },
            model_dir=tmp_path,
        )
        m = read_manifest("EURUSD", model_dir=tmp_path)
        assert m["instrument"] == "EURUSD"
        assert m["ob_max_distance_pips"] == 30.0
        assert m["retrain_report"] == "/tmp/report.json"


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


class TestValidateManifestAgainstEnv:

    def test_no_manifest_no_drift(self, monkeypatch):
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "30.0")
        assert validate_manifest_against_env(None, instrument="EURUSD") is None

    def test_matching_env_no_drift(self, monkeypatch):
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "30.0")
        manifest = {"ob_max_distance_pips": 30.0}
        assert validate_manifest_against_env(manifest, instrument="EURUSD") is None

    def test_mismatched_env_returns_drift_message(self, monkeypatch):
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "50.0")
        manifest = {"ob_max_distance_pips": 30.0}
        msg = validate_manifest_against_env(manifest, instrument="EURUSD")
        assert msg is not None
        assert "EURUSD" in msg
        assert "30" in msg
        assert "50" in msg
        assert "PHASE_7A" in msg

    def test_string_env_value_parsed(self, monkeypatch):
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "30.0")
        manifest = {"ob_max_distance_pips": "30.0"}  # stored as str
        assert validate_manifest_against_env(manifest, instrument="EURUSD") is None

    def test_unparseable_env_returns_message(self, monkeypatch):
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "garbage")
        manifest = {"ob_max_distance_pips": 30.0}
        msg = validate_manifest_against_env(manifest, instrument="EURUSD")
        assert msg is not None
        assert "garbage" in msg

    def test_unparseable_manifest_returns_message(self, monkeypatch):
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "30.0")
        manifest = {"ob_max_distance_pips": "not-a-float"}
        msg = validate_manifest_against_env(manifest, instrument="EURUSD")
        assert msg is not None
        assert "malformed" in msg

    def test_missing_field_in_manifest_no_drift(self, monkeypatch):
        # Pre-Phase-7A manifest format (legacy, without the field).
        # We tolerate this so the rollout is backward compatible.
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "30.0")
        manifest = {"some_other_field": True}
        assert validate_manifest_against_env(manifest, instrument="EURUSD") is None

    def test_default_env_when_unset(self, monkeypatch):
        monkeypatch.delenv("PRISM_OB_MAX_DISTANCE_PIPS", raising=False)
        manifest = {"ob_max_distance_pips": 30.0}
        # Default is 30.0 — no drift
        assert validate_manifest_against_env(manifest, instrument="EURUSD") is None

    def test_tiny_float_difference_tolerated(self, monkeypatch):
        # Within 1e-6 tolerance — accumulator drift, not a real change
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "30.0000001")
        manifest = {"ob_max_distance_pips": 30.0}
        assert validate_manifest_against_env(manifest, instrument="EURUSD") is None


# ---------------------------------------------------------------------------
# PRISMPredictor integration — strict-mode env switch
# ---------------------------------------------------------------------------


def _train_fixture_models(model_dir: Path, instrument: str = "EURUSD") -> None:
    """Build the four joblib artifacts that ``PRISMPredictor`` requires.

    Uses tiny synthetic models — same pattern as ``tests/test_model.py``.
    """
    import joblib
    import lightgbm as lgb
    import numpy as np
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y_mapped = rng.integers(0, 3, size=80)

    clf_xgb = xgb.XGBClassifier(
        n_estimators=3, max_depth=2, random_state=42,
        objective="multi:softprob", num_class=3, verbosity=0,
    )
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=3, max_depth=2, random_state=42,
        objective="multiclass", num_class=3, verbose=-1,
    )
    reg = xgb.XGBRegressor(n_estimators=3, random_state=42, verbosity=0)
    clf_rf = RandomForestClassifier(n_estimators=3, random_state=42)

    clf_xgb.fit(X, y_mapped)
    clf_lgb.fit(X, y_mapped)
    reg.fit(X, rng.uniform(5, 50, 80))
    clf_rf.fit(X, y_mapped)

    joblib.dump(clf_xgb, model_dir / f"layer1_xgb_{instrument}.joblib")
    joblib.dump(clf_lgb, model_dir / f"layer1_lgbm_{instrument}.joblib")
    joblib.dump(reg,     model_dir / f"layer2_reg_{instrument}.joblib")
    joblib.dump(clf_rf,  model_dir / f"layer3_rf_{instrument}.joblib")


class TestPredictorManifestIntegration:
    """End-to-end coverage of how ``PRISMPredictor.__init__`` interacts
    with the manifest. Uses tiny fixture models so the load path runs
    in milliseconds."""

    def test_predictor_warns_on_drift_by_default(self, tmp_path, monkeypatch, caplog):
        import prism.model.predict as predict_mod

        _train_fixture_models(tmp_path)
        write_manifest("EURUSD", ob_max_distance_pips=30.0, model_dir=tmp_path)

        monkeypatch.setattr(predict_mod, "MODEL_DIR", tmp_path)
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "50.0")
        monkeypatch.delenv("PRISM_OB_MAX_DISTANCE_PIPS_STRICT", raising=False)

        with caplog.at_level("WARNING", logger="prism.model.predict"):
            predictor = predict_mod.PRISMPredictor("EURUSD")

        # Predictor still constructs OK — drift is non-fatal by default
        assert predictor._manifest is not None
        assert any("drift" in r.message.lower() for r in caplog.records)

    def test_predictor_raises_on_drift_in_strict_mode(self, tmp_path, monkeypatch):
        import prism.model.predict as predict_mod

        _train_fixture_models(tmp_path)
        write_manifest("EURUSD", ob_max_distance_pips=30.0, model_dir=tmp_path)

        monkeypatch.setattr(predict_mod, "MODEL_DIR", tmp_path)
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "50.0")
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS_STRICT", "1")

        with pytest.raises(RuntimeError, match="drift"):
            predict_mod.PRISMPredictor("EURUSD")

    def test_predictor_loads_clean_when_no_drift(self, tmp_path, monkeypatch):
        import prism.model.predict as predict_mod

        _train_fixture_models(tmp_path)
        write_manifest("EURUSD", ob_max_distance_pips=30.0, model_dir=tmp_path)

        monkeypatch.setattr(predict_mod, "MODEL_DIR", tmp_path)
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "30.0")

        predictor = predict_mod.PRISMPredictor("EURUSD")
        assert predictor._manifest is not None
        assert predictor._manifest["ob_max_distance_pips"] == 30.0

    def test_predictor_loads_when_no_manifest_present(self, tmp_path, monkeypatch):
        """Backward compatibility: legacy models pre-Phase-7A have no
        manifest. Predictor must load cleanly without warning, even if
        strict mode is on (no manifest → nothing to check)."""
        import prism.model.predict as predict_mod

        _train_fixture_models(tmp_path)
        # No manifest written

        monkeypatch.setattr(predict_mod, "MODEL_DIR", tmp_path)
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS_STRICT", "1")

        predictor = predict_mod.PRISMPredictor("EURUSD")
        assert predictor._manifest is None
