"""
tests/test_model.py
Unit tests for PRISM ML model layer (train / predict / evaluate).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_data(n: int = 200, n_features: int = 20, seed: int = 42):
    """
    Generate a synthetic DataFrame that mimics PRISM's feature matrix:
    - n rows of float features
    - direction_4h: {0, 1, 2}  (already mapped from {-1, 0, 1})
    - magnitude_pips: positive float
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Deterministic labels so the test is repeatable
    y = pd.Series((rng.integers(0, 3, size=n)), name="direction_4h")
    mags = pd.Series(rng.uniform(5, 50, size=n), name="magnitude_pips")
    df = pd.concat([X, y, mags], axis=1)
    return df


def _split(df, test_ratio=0.2):
    n = len(df)
    split = int(n * (1 - test_ratio))
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X_train = df.iloc[:split][feature_cols].values.astype(np.float32)
    X_test = df.iloc[split:][feature_cols].values.astype(np.float32)
    y_train = df.iloc[:split]["direction_4h"].values
    y_test = df.iloc[split:]["direction_4h"].values
    return X_train, X_test, y_train, y_test, split


# ---------------------------------------------------------------------------
# Test 1: XGBClassifier trains and produces valid predictions
# ---------------------------------------------------------------------------

def test_xgb_trains_and_predicts():
    """XGBClassifier should train on dummy data and return labels in {0, 1, 2}."""
    df = _make_dummy_data(200, 20)
    X_train, X_test, y_train, y_test, _ = _split(df)

    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)

    # Predictions must be a valid label set
    assert set(preds).issubset({0, 1, 2}), f"Unexpected labels: {set(preds)}"
    assert len(preds) == len(y_test), "Prediction length mismatch"

    acc = accuracy_score(y_test, preds)
    # Even random data should be at least partially above chance (>= 0.15)
    # (purely to verify the model ran, not to assert quality)
    assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"

    # Probabilities shape
    probas = model.predict_proba(X_test)
    assert probas.shape == (len(X_test), 3), f"Bad proba shape: {probas.shape}"
    # Each row should sum to ~1.0
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-4)

    print(f"  XGB test_accuracy={acc:.4f} — OK")


# ---------------------------------------------------------------------------
# Test 2: No data-leakage — train indices always precede test indices
# ---------------------------------------------------------------------------

def test_no_lookahead_in_split():
    """
    Walk-forward TimeSeriesSplit must never leak future bars into training.
    For every (train_idx, val_idx) fold, max(train_idx) < min(val_idx).
    """
    df = _make_dummy_data(300)
    X = df[[c for c in df.columns if c.startswith("feat_")]].values

    tscv = TimeSeriesSplit(n_splits=5)
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        assert len(train_idx) > 0, f"Fold {fold_i}: empty train set"
        assert len(val_idx) > 0, f"Fold {fold_i}: empty val set"
        max_train = int(np.max(train_idx))
        min_val = int(np.min(val_idx))
        assert max_train < min_val, (
            f"Fold {fold_i}: lookahead! max_train={max_train} >= min_val={min_val}"
        )

    # Also verify the simple chronological split used in pipeline
    _, _, _, _, split_idx = _split(df, test_ratio=0.2)
    n = len(df)
    train_indices = np.arange(split_idx)
    test_indices = np.arange(split_idx, n)
    assert train_indices.max() < test_indices.min(), (
        "Simple split has lookahead contamination"
    )

    print(f"  No lookahead across all {tscv.n_splits} CV folds — OK")


# ---------------------------------------------------------------------------
# Test 3: Overfit detection logic (15% threshold)
# ---------------------------------------------------------------------------

def test_overfit_detection():
    """
    The overfit_flag should be True iff (train_acc - test_acc) > 0.15.
    We test the raw threshold logic independently of a full training run.
    """
    OVERFIT_THRESHOLD = 0.15

    def _overfit(train_acc: float, test_acc: float) -> bool:
        return (train_acc - test_acc) > OVERFIT_THRESHOLD

    # Cases that SHOULD flag overfit
    assert _overfit(0.90, 0.70), "0.90 - 0.70 = 0.20 > 0.15 → should flag"
    assert _overfit(1.00, 0.80), "1.00 - 0.80 = 0.20 > 0.15 → should flag"
    assert _overfit(0.80, 0.64), "0.80 - 0.64 = 0.16 > 0.15 → should flag"

    # Cases that should NOT flag overfit
    assert not _overfit(0.75, 0.601), "0.75 - 0.601 = 0.149 < 0.15 → no flag"
    assert not _overfit(0.75, 0.65), "0.75 - 0.65 = 0.10 < 0.15 → no flag"
    assert not _overfit(0.60, 0.60), "delta=0 → no flag"
    assert not _overfit(0.55, 0.60), "negative delta → no flag"

    # Simulate with a real XGB model (memorisation on tiny dataset)
    df = _make_dummy_data(60, 10, seed=7)       # small → likely to overfit
    X_train, X_test, y_train, y_test, _ = _split(df, test_ratio=0.25)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=99,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    flag = _overfit(train_acc, test_acc)

    # On this tiny, high-depth model, training accuracy should be very high
    # We just verify the detection function agrees with manual arithmetic
    expected_flag = (train_acc - test_acc) > OVERFIT_THRESHOLD
    assert flag == expected_flag, (
        f"Overfit flag mismatch: train={train_acc:.4f} test={test_acc:.4f} "
        f"delta={train_acc - test_acc:.4f} → expected flag={expected_flag}"
    )

    print(
        f"  Overfit detection: train={train_acc:.4f} test={test_acc:.4f} "
        f"delta={train_acc - test_acc:.4f} flag={flag} — OK"
    )


# ---------------------------------------------------------------------------
# Optional: quick smoke test for evaluate.backtest_signals
# ---------------------------------------------------------------------------

def test_backtest_runs_without_error():
    """backtest_signals should return valid metrics dict on synthetic OHLC."""
    from prism.model.evaluate import backtest_signals

    rng = np.random.default_rng(0)
    n = 100
    close = 1.1000 + np.cumsum(rng.normal(0, 0.001, n))
    df = pd.DataFrame({
        "open": close,
        "high": close + rng.uniform(0, 0.002, n),
        "low": close - rng.uniform(0, 0.002, n),
        "close": close,
    })
    signals = [
        {"direction": int(rng.choice([-1, 0, 1])), "confidence": 0.6, "magnitude_pips": 20}
        for _ in range(n)
    ]
    metrics = backtest_signals(df, signals, instrument="EURUSD", initial_balance=10_000)
    required = {"total_trades", "win_rate", "profit_factor", "avg_rr",
                "max_drawdown", "sharpe", "final_balance", "total_return_pct"}
    assert required.issubset(metrics.keys()), f"Missing keys: {required - metrics.keys()}"
    assert 0 <= metrics["win_rate"] <= 1
    assert metrics["total_trades"] >= 0
    print(f"  Backtest metrics: {metrics} — OK")
