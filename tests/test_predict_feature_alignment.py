"""
tests/test_predict_feature_alignment.py

Regression test for the predict-time feature-shape mismatch crash that
was crash-looping the production runner with::

    Feature shape mismatch, expected: 37, got 38

The trainer now persists the ordered feature column list to
``models/feature_cols_<INSTRUMENT>.json`` and ``PRISMPredictor`` uses
that sidecar to project the live frame onto the trained schema before
inference. Drift becomes a logged WARNING, not a hard crash.

Covers:

* End-to-end: train a tiny model with the trainer-public helper that
  writes the sidecar, then load via ``PRISMPredictor`` and verify the
  schema is loaded and used.
* Out-of-order columns + an unexpected extra column: predict succeeds,
  WARNING logged listing the extra column.
* Missing column: predict succeeds with zero-fill, WARNING logged
  listing the missing column.
* Legacy joblibs without a sidecar: predictor refuses to load with a
  clear retrain instruction.
* ``missing_model_files`` flags the missing sidecar.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prism.model.predict as predict_mod
import prism.model.train as train_mod


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _train_tiny_models(tmpdir: Path, instrument: str, feature_cols: list[str]) -> None:
    """Train a tiny set of joblibs + write the feature_cols sidecar.

    Mirrors what ``PRISMTrainer.train_all_layers`` produces, but runs in
    seconds and only depends on the columns we hand in. Avoids the
    full pipeline stack (Tiingo / FRED / quiver) so the test is hermetic.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame(
        rng.standard_normal((n, len(feature_cols))),
        columns=feature_cols,
    )
    y = pd.Series(rng.integers(0, 3, size=n))
    mags = rng.uniform(5, 50, size=n)

    clf_xgb = xgb.XGBClassifier(
        n_estimators=5, max_depth=2, random_state=0,
        objective="multi:softprob", num_class=3, verbosity=0,
    )
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=5, max_depth=2, random_state=0,
        objective="multiclass", num_class=3, verbose=-1,
    )
    reg = xgb.XGBRegressor(n_estimators=5, random_state=0, verbosity=0)
    clf_rf = RandomForestClassifier(n_estimators=5, random_state=0)

    clf_xgb.fit(X, y)
    clf_lgb.fit(X, y)
    reg.fit(X, mags)
    clf_rf.fit(X, y)

    joblib.dump(clf_xgb, tmpdir / f"layer1_xgb_{instrument}.joblib")
    joblib.dump(clf_lgb, tmpdir / f"layer1_lgbm_{instrument}.joblib")
    joblib.dump(reg,     tmpdir / f"layer2_reg_{instrument}.joblib")
    joblib.dump(clf_rf,  tmpdir / f"layer3_rf_{instrument}.joblib")

    # Use the public helper so we test the same path retrain.py uses.
    train_mod.write_feature_cols(instrument, feature_cols, model_dir=tmpdir)


@pytest.fixture
def tiny_models_dir(monkeypatch):
    """Yields a Path to a tmpdir populated with tiny models + feature_cols
    sidecar for ``EURUSD``. ``predict_mod.MODEL_DIR`` is monkey-patched
    to point at it for the duration of the test.
    """
    feature_cols = ["A", "B", "C"]
    instrument = "EURUSD"
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        _train_tiny_models(tmpdir, instrument, feature_cols)
        monkeypatch.setattr(predict_mod, "MODEL_DIR", tmpdir)
        yield tmpdir, instrument, feature_cols


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predictor_loads_feature_cols_sidecar(tiny_models_dir):
    """PRISMPredictor.__init__ must load the feature_cols sidecar."""
    tmpdir, instrument, feature_cols = tiny_models_dir

    predictor = predict_mod.PRISMPredictor(instrument)

    assert predictor._feature_cols == feature_cols, (
        f"Expected feature_cols {feature_cols}, got {predictor._feature_cols}"
    )


def test_predict_handles_out_of_order_plus_extra_column(tiny_models_dir, caplog):
    """Predict must NOT crash when columns are out of order plus an
    unexpected extra column. A WARNING must be logged listing the
    extra column. Result arrays must still come back per-row.
    """
    tmpdir, instrument, feature_cols = tiny_models_dir
    rng = np.random.default_rng(7)

    # Reorder the trained columns and add an unexpected extra "D"
    n = 6
    drifted = pd.DataFrame(
        rng.standard_normal((n, 4)),
        columns=["C", "B", "A", "D"],  # out-of-order + extra D
    )

    predictor = predict_mod.PRISMPredictor(instrument)

    with caplog.at_level(logging.WARNING, logger="prism.model.predict"):
        result = predictor.predict(drifted)

    # No crash, full per-row result
    assert len(result["direction"]) == n
    assert len(result["magnitude_pips"]) == n
    assert set(result["direction"]).issubset({-1, 0, 1})

    # WARNING was logged and mentions D as an extra column
    drift_msgs = [
        rec.getMessage() for rec in caplog.records
        if rec.levelno == logging.WARNING and "feature drift" in rec.getMessage()
    ]
    assert drift_msgs, f"Expected drift WARNING, got: {[r.getMessage() for r in caplog.records]}"
    msg = drift_msgs[0]
    assert "'D'" in msg or "\"D\"" in msg or "D" in msg, (
        f"Drift warning does not mention extra column 'D': {msg}"
    )
    assert "extra_in_live" in msg


def test_predict_handles_missing_column(tiny_models_dir, caplog):
    """Predict must NOT crash when a trained column is missing from the
    live frame. A WARNING must be logged naming the missing column.
    The missing column is zero-filled by ``DataFrame.reindex``.
    """
    tmpdir, instrument, feature_cols = tiny_models_dir
    rng = np.random.default_rng(9)

    n = 6
    drifted = pd.DataFrame(
        rng.standard_normal((n, 2)),
        columns=["A", "C"],  # column B missing
    )

    predictor = predict_mod.PRISMPredictor(instrument)

    with caplog.at_level(logging.WARNING, logger="prism.model.predict"):
        result = predictor.predict(drifted)

    assert len(result["direction"]) == n

    drift_msgs = [
        rec.getMessage() for rec in caplog.records
        if rec.levelno == logging.WARNING and "feature drift" in rec.getMessage()
    ]
    assert drift_msgs, "Expected drift WARNING"
    msg = drift_msgs[0]
    assert "missing_in_live" in msg
    assert "B" in msg, f"Drift warning does not mention missing column B: {msg}"


def test_drift_warning_logged_only_once(tiny_models_dir, caplog):
    """The drift warning is rate-limited to one per predictor session."""
    tmpdir, instrument, feature_cols = tiny_models_dir
    rng = np.random.default_rng(11)
    drifted = pd.DataFrame(
        rng.standard_normal((4, 4)),
        columns=["C", "B", "A", "D"],
    )
    predictor = predict_mod.PRISMPredictor(instrument)

    with caplog.at_level(logging.WARNING, logger="prism.model.predict"):
        predictor.predict(drifted)
        predictor.predict(drifted)
        predictor.predict(drifted)

    drift_msgs = [
        rec.getMessage() for rec in caplog.records
        if rec.levelno == logging.WARNING and "feature drift" in rec.getMessage()
    ]
    assert len(drift_msgs) == 1, (
        f"Expected exactly 1 drift warning across 3 predict() calls, "
        f"got {len(drift_msgs)}"
    )


def test_predict_aligned_columns_no_warning(tiny_models_dir, caplog):
    """When the live frame already matches the trained schema (same
    columns, same order), no drift warning should fire.
    """
    tmpdir, instrument, feature_cols = tiny_models_dir
    rng = np.random.default_rng(13)
    aligned = pd.DataFrame(
        rng.standard_normal((4, 3)),
        columns=feature_cols,
    )
    predictor = predict_mod.PRISMPredictor(instrument)

    with caplog.at_level(logging.WARNING, logger="prism.model.predict"):
        result = predictor.predict(aligned)

    drift_msgs = [
        rec.getMessage() for rec in caplog.records
        if rec.levelno == logging.WARNING and "feature drift" in rec.getMessage()
    ]
    assert not drift_msgs, f"Unexpected drift warning on aligned frame: {drift_msgs}"
    assert len(result["direction"]) == 4


def test_legacy_joblibs_without_sidecar_fail_loud(monkeypatch):
    """Legacy joblibs predating this fix don't have a feature_cols
    sidecar. PRISMPredictor must refuse to load and tell the operator
    to retrain. Silent legacy mode is NOT supported \u2014 it would
    silently mis-predict.
    """
    feature_cols = ["A", "B", "C"]
    instrument = "EURUSD"
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        _train_tiny_models(tmpdir, instrument, feature_cols)
        # Remove the feature_cols sidecar to simulate a legacy install
        (tmpdir / f"feature_cols_{instrument}.json").unlink()

        monkeypatch.setattr(predict_mod, "MODEL_DIR", tmpdir)

        with pytest.raises(FileNotFoundError) as exc_info:
            predict_mod.PRISMPredictor(instrument)

        msg = str(exc_info.value)
        assert "feature_cols" in msg
        assert "retrain" in msg.lower()
        assert instrument in msg


def test_missing_model_files_flags_missing_sidecar(monkeypatch):
    """``missing_model_files`` must include the feature_cols sidecar in
    its check so the runner refuses to start without it.
    """
    feature_cols = ["A", "B", "C"]
    instrument = "EURUSD"
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        _train_tiny_models(tmpdir, instrument, feature_cols)

        # All present \u2192 nothing missing
        missing = predict_mod.missing_model_files([instrument], model_dir=tmpdir)
        assert missing == [], f"Expected no missing files, got {missing}"

        # Remove the sidecar \u2192 it should now show up as missing
        (tmpdir / f"feature_cols_{instrument}.json").unlink()
        missing = predict_mod.missing_model_files([instrument], model_dir=tmpdir)
        assert any(
            p.name == f"feature_cols_{instrument}.json" for p in missing
        ), f"Expected feature_cols sidecar in missing list, got {missing}"


def test_feature_cols_sidecar_round_trip(tmp_path):
    """Round-trip: write_feature_cols \u2192 read_feature_cols should
    return the same ordered list."""
    cols = ["alpha", "beta", "gamma", "delta"]
    instrument = "GBPUSD"

    train_mod.write_feature_cols(instrument, cols, model_dir=tmp_path)
    sidecar_path = tmp_path / f"feature_cols_{instrument}.json"
    assert sidecar_path.exists()

    # Validate file content shape
    payload = json.loads(sidecar_path.read_text())
    assert payload["instrument"] == instrument
    assert payload["n_features"] == len(cols)
    assert payload["feature_cols"] == cols

    loaded = predict_mod.read_feature_cols(instrument, model_dir=tmp_path)
    assert loaded == cols, f"Round-trip mismatch: wrote {cols}, read {loaded}"
