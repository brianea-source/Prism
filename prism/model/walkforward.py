"""Phase 7.A walk-forward harness.

Implements the rolling-window walk-forward evaluator specified in
``docs/PHASE_7A_SCOPE.md`` §5: 18-month windows, 80/20 chronological
train/test split inside each window, 1-month step size, 24 anchored
folds spanning ~2 years of out-of-sample test starts.

Why a separate harness instead of reusing ``PRISMFeaturePipeline``'s
single 80/20 split (``pipeline.py:split_train_test``):

* A single chronological split conflates "did the model learn?" with
  "did the features happen to help in the trailing 6 months?". Walk-
  forward decouples the two — feature lift becomes the median across
  folds, regime sensitivity becomes the min-max envelope.
* Phase 7.A's promotion gates (1-3) are explicitly stated in terms of
  walk-forward statistics. Gate 4 (SHAP stability) and gate 5
  (live-vs-historical drift) live elsewhere; this harness owns the
  three trade-statistics gates.

Independence caveat carried forward from the scope doc: with a 1-month
step on 3.6-month test windows, adjacent test windows overlap by 2.6
months — the 24 fold results are NOT 24 independent samples. Treat the
median as a *stability* estimator, not a confidence interval.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — defaults match scope §5 spec exactly. Centralised so test
# fixtures can shorten the window without redefining everything.
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_MONTHS: int = 18
DEFAULT_STEP_MONTHS: int = 1
DEFAULT_TRAIN_RATIO: float = 0.8

#: Acceptance gates per PHASE_7A_SCOPE.md §6 (gates 1-3). Gate 4 (SHAP)
#: and gate 5 (drift) are evaluated by separate machinery.
GATE_F1_NEW_GTE_BASELINE = "median_f1_new >= median_f1_baseline"
GATE_SHARPE_NEW_GTE_BASELINE_X_095 = "median_sharpe_new >= 0.95 * median_sharpe_baseline"
GATE_MAXDD_NEW_LTE_BASELINE_X_110 = "median_max_dd_new <= 1.10 * median_max_dd_baseline"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Per-fold metrics. ``f1`` is the macro-averaged classifier F1
    against the integer direction label; ``sharpe`` and ``max_drawdown``
    come from the trading-side ``backtest_signals`` evaluator so the
    metrics match the live performance dashboard exactly.

    Timestamps are stored as ISO strings for clean JSON serialisation —
    asdict() on a tz-aware pd.Timestamp is fragile across pandas
    versions.
    """
    fold_idx: int
    window_start: str
    window_end: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    f1: float
    sharpe: float
    max_drawdown: float


@dataclass
class WalkforwardResult:
    instrument: str
    feature_set: str  # "baseline" / "phase7a" / arbitrary tag
    n_folds: int
    folds: list[FoldResult] = field(default_factory=list)
    median_f1: float = float("nan")
    min_f1: float = float("nan")
    max_f1: float = float("nan")
    median_sharpe: float = float("nan")
    min_sharpe: float = float("nan")
    max_sharpe: float = float("nan")
    median_max_drawdown: float = float("nan")
    min_max_drawdown: float = float("nan")
    max_max_drawdown: float = float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["folds"] = [asdict(f) for f in self.folds]
        return d


# ---------------------------------------------------------------------------
# Fold boundary generator (no model required — pure indexing logic)
# ---------------------------------------------------------------------------


def generate_fold_boundaries(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "datetime",
    window_months: int = DEFAULT_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    n_folds: Optional[int] = None,
) -> list[dict]:
    """Compute the (window, train, test) timestamp triples for the walk-forward.

    Returns one dict per fold with keys ``window_start``, ``window_end``,
    ``train_end``, ``test_start``, ``test_end`` — all ``pd.Timestamp``.
    The total window length is ``window_months``, the test segment
    starts at ``window_start + window_months * train_ratio`` and runs to
    ``window_end``. Successive folds slide the anchor by ``step_months``.

    When ``n_folds`` is None, the count is inferred as
    ``floor((data_span - window_months) / step_months) + 1``, capped at
    24 to match the scope spec.

    Args:
        df: Must contain ``timestamp_col`` with monotone-increasing UTC
            timestamps. Sorting is enforced internally.
        n_folds: Override the inferred count. Useful for unit tests
            that supply 24-month synthetic data and want to assert
            fold count.

    Raises:
        ValueError: If the data span is shorter than one window.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if window_months <= 0 or step_months <= 0:
        raise ValueError("window_months and step_months must be > 0")

    ts = pd.to_datetime(df[timestamp_col]).sort_values().reset_index(drop=True)
    if ts.empty:
        raise ValueError("generate_fold_boundaries: empty timestamp series")

    data_start = ts.iloc[0]
    data_end = ts.iloc[-1]
    data_span_months = (data_end.year - data_start.year) * 12 + (
        data_end.month - data_start.month
    )

    if data_span_months < window_months:
        raise ValueError(
            f"data span {data_span_months}mo < window_months {window_months}; "
            "cannot run walk-forward",
        )

    inferred = max(0, (data_span_months - window_months) // step_months) + 1
    if n_folds is None:
        n = min(inferred, 24)
    else:
        n = min(n_folds, inferred)
    if n <= 0:
        raise ValueError("walk-forward computed 0 folds — input span too short")

    folds: list[dict] = []
    window_offset = pd.DateOffset(months=window_months)
    train_offset = pd.DateOffset(
        months=int(np.floor(window_months * train_ratio)),
    )
    step_offset = pd.DateOffset(months=step_months)

    for i in range(n):
        window_start = data_start + (step_offset * i)
        window_end = window_start + window_offset
        train_end = window_start + train_offset
        if window_end > data_end:
            # Truncate the trailing fold rather than dropping it — gives
            # the caller something to work with on barely-sufficient data.
            window_end = data_end
        folds.append({
            "window_start": window_start,
            "window_end": window_end,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": window_end,
        })
    return folds


# ---------------------------------------------------------------------------
# Default fit/predict — lightgbm classifier on the integer direction label
# ---------------------------------------------------------------------------


FitPredictFn = Callable[
    [pd.DataFrame, pd.Series, pd.DataFrame],
    np.ndarray,
]
"""Signature: ``(X_train, y_train, X_test) -> y_pred (1D ndarray)``.

The default :func:`_default_fit_predict` returns class predictions
(integers in ``{-1, 0, 1}``) so the harness can compute F1 directly.
Custom callables can return any integer encoding — the harness only
requires that ``y_test`` and ``y_pred`` share an alphabet.
"""


def _default_fit_predict(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
) -> np.ndarray:
    """LightGBM multiclass classifier, deterministic seed.

    Imported lazily so test environments without lightgbm can still
    drive the harness with a custom ``fit_predict_fn``.
    """
    from lightgbm import LGBMClassifier

    # Map {-1, 0, 1} to {0, 1, 2} for LightGBM (it requires non-negative
    # class labels), then map back at predict time.
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_map = {0: -1, 1: 0, 2: 1}

    y_train_mapped = y_train.map(label_map).fillna(1).astype(int)
    if y_train_mapped.nunique() < 2:
        # Single-class fold — model can't learn; return all-neutral.
        # Caller's F1 will be poor but no exception, so the fold counts.
        return np.zeros(len(X_test), dtype=int)

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train.fillna(0), y_train_mapped)
    pred_mapped = model.predict(X_test.fillna(0))
    return np.array([inv_map.get(int(p), 0) for p in pred_mapped])


# ---------------------------------------------------------------------------
# Backtest hook for trade-side metrics (Sharpe, MaxDD)
# ---------------------------------------------------------------------------


def _signals_from_predictions(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
) -> list[dict]:
    """Convert per-bar classifier predictions into ``backtest_signals``'s
    expected shape.

    ``backtest_signals`` requires one record per bar in ``test_df`` (it
    walks them in lockstep) with integer ``direction`` in ``{-1, 0, 1}``
    and ``confidence`` in ``[0, 1]``. Bars where the model predicts no
    trade get ``direction=0`` rather than being dropped — dropping rows
    would misalign the lookforward window with the OHLC frame.

    Magnitude defaults to 30 pips on bars without a per-bar
    ``magnitude_pips`` column so the backtester applies the same SL/TP
    geometry across folds — varying it per fold would confound the
    feature-lift comparison.
    """
    n = len(test_df)
    if n == 0:
        return []
    n_pred = min(len(y_pred), n)

    has_magnitude = "magnitude_pips" in test_df.columns
    magnitudes = (
        test_df["magnitude_pips"].fillna(30.0).to_numpy()
        if has_magnitude
        else np.full(n, 30.0)
    )

    signals: list[dict] = []
    for i in range(n):
        pred = int(y_pred[i]) if i < n_pred else 0
        if pred not in (-1, 0, 1):
            pred = 0
        signals.append({
            "direction": pred,
            "confidence": 0.7,
            "magnitude_pips": float(magnitudes[i]),
        })
    return signals


def _fold_trade_metrics(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    instrument: str,
) -> tuple[float, float]:
    """Run a Sharpe + MaxDD backtest on the predictions.

    Reuses ``prism.model.evaluate.backtest_signals`` so the per-fold
    metrics align with the live performance dashboard. Returns
    ``(sharpe, max_drawdown_magnitude)``. The magnitude is the absolute
    value of ``backtest_signals``'s ``max_drawdown`` (which it returns
    as a non-positive ratio); using positive magnitudes throughout
    the harness makes the gate-3 multiplicative comparison
    (``new <= 1.10 * baseline``) read naturally.

    Empty-signals folds (rare — only when ``test_df`` is empty) get
    ``(0.0, 0.0)``.
    """
    signals = _signals_from_predictions(test_df, y_pred)
    if not signals:
        return 0.0, 0.0

    try:
        from prism.model.evaluate import backtest_signals
        result = backtest_signals(test_df, signals, instrument=instrument)
        sharpe = float(result.get("sharpe", 0.0))
        # backtest_signals returns max_drawdown as a non-positive ratio;
        # convert to a positive magnitude so the gate comparison is
        # invariant to sign convention drift in evaluate.py.
        raw_dd = float(result.get("max_drawdown", 0.0))
        return sharpe, abs(raw_dd)
    except Exception as exc:
        # Defensive — backtest_signals can raise on malformed input
        # (e.g. tiny test segments). Don't let one fold kill the
        # harness; log loud and return neutral metrics.
        logger.warning(
            "backtest_signals failed on fold (n_signals=%d): %s",
            len(signals), exc,
        )
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


def run_walkforward(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    instrument: str,
    feature_set: str = "baseline",
    label_col: str = "direction_fwd_4",
    timestamp_col: str = "datetime",
    window_months: int = DEFAULT_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    n_folds: Optional[int] = None,
    fit_predict_fn: Optional[FitPredictFn] = None,
) -> WalkforwardResult:
    """Run the walk-forward, return a structured result.

    Args:
        df: Feature matrix with ``timestamp_col``, ``label_col``, and
            every column in ``feature_cols``. Sorting is enforced.
        feature_cols: Columns to feed the model. Use
            ``ICTFeatureEngineer().feature_columns()`` to get the
            Phase 7.A subset.
        instrument: For the backtest hook.
        feature_set: Free-form tag persisted into the result; canonical
            choices are ``"baseline"`` and ``"phase7a"``.
        label_col: Integer-encoded direction column. Default matches
            ``PRISMFeaturePipeline._engineer_features``.
        n_folds: Override the inferred fold count.
        fit_predict_fn: Custom model callable. Default lazy-imports
            LightGBM.

    Returns:
        :class:`WalkforwardResult` with per-fold breakdowns and the
        median/min/max aggregates required by gates 1-3.
    """
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("run_walkforward: feature_cols cannot be empty")

    required = {timestamp_col, label_col, *feature_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    folds_meta = generate_fold_boundaries(
        df,
        timestamp_col=timestamp_col,
        window_months=window_months,
        step_months=step_months,
        train_ratio=train_ratio,
        n_folds=n_folds,
    )

    fit_predict = fit_predict_fn or _default_fit_predict

    fold_results: list[FoldResult] = []
    for idx, fold in enumerate(folds_meta):
        ts = df[timestamp_col]
        in_window = (ts >= fold["window_start"]) & (ts <= fold["window_end"])
        is_train = in_window & (ts < fold["train_end"])
        is_test = in_window & (ts >= fold["test_start"])

        train_slice = df[is_train]
        test_slice = df[is_test]

        if len(train_slice) < 50 or len(test_slice) < 10:
            logger.warning(
                "fold %d: insufficient rows (train=%d, test=%d) — skipping",
                idx, len(train_slice), len(test_slice),
            )
            continue

        X_train = train_slice[feature_cols]
        X_test = test_slice[feature_cols]
        y_train = train_slice[label_col]
        y_test = test_slice[label_col]

        try:
            y_pred = fit_predict(X_train, y_train, X_test)
        except Exception as exc:
            logger.error("fold %d: fit_predict raised %s — skipping", idx, exc)
            continue

        try:
            f1 = float(f1_score(
                y_test.astype(int).values,
                np.asarray(y_pred).astype(int),
                average="macro",
                labels=[-1, 0, 1],
                zero_division=0,
            ))
        except Exception as exc:
            logger.warning("fold %d: f1 compute failed: %s", idx, exc)
            f1 = 0.0

        sharpe, max_dd = _fold_trade_metrics(
            test_slice, y_pred, instrument=instrument,
        )

        fold_results.append(FoldResult(
            fold_idx=idx,
            window_start=fold["window_start"].isoformat(),
            window_end=fold["window_end"].isoformat(),
            train_end=fold["train_end"].isoformat(),
            test_start=fold["test_start"].isoformat(),
            test_end=fold["test_end"].isoformat(),
            n_train=len(train_slice),
            n_test=len(test_slice),
            f1=f1,
            sharpe=sharpe,
            max_drawdown=max_dd,
        ))

    result = WalkforwardResult(
        instrument=instrument,
        feature_set=feature_set,
        n_folds=len(fold_results),
        folds=fold_results,
    )

    if fold_results:
        f1_arr = np.array([f.f1 for f in fold_results])
        sh_arr = np.array([f.sharpe for f in fold_results])
        dd_arr = np.array([f.max_drawdown for f in fold_results])
        result.median_f1 = float(np.median(f1_arr))
        result.min_f1 = float(np.min(f1_arr))
        result.max_f1 = float(np.max(f1_arr))
        result.median_sharpe = float(np.median(sh_arr))
        result.min_sharpe = float(np.min(sh_arr))
        result.max_sharpe = float(np.max(sh_arr))
        result.median_max_drawdown = float(np.median(dd_arr))
        result.min_max_drawdown = float(np.min(dd_arr))
        result.max_max_drawdown = float(np.max(dd_arr))

    return result


# ---------------------------------------------------------------------------
# Acceptance gates 1-3 (PHASE_7A_SCOPE.md §6)
# ---------------------------------------------------------------------------


def acceptance_decision(
    baseline: WalkforwardResult,
    candidate: WalkforwardResult,
) -> dict[str, Any]:
    """Evaluate gates 1, 2, 3 of the Phase 7.A acceptance criteria.

    Gates 4 (SHAP stability) and 5 (live-vs-historical drift) live in
    ``prism/audit/smart_money_export.py`` and the SHAP harness — this
    function only owns the trade-statistics gates.

    Returns a dict with keys ``passed``, ``gates``, ``rationale``. The
    ``gates`` sub-dict has one entry per gate with ``passed: bool`` +
    ``baseline``, ``candidate`` numeric values for transparency.

    Convention: ``max_drawdown`` is reported as a *positive* number by
    ``backtest_signals`` (the magnitude of the worst peak-to-trough
    decline), so the gate compares ``new <= 1.10 * baseline``.
    """
    g1_passed = candidate.median_f1 >= baseline.median_f1
    g2_passed = candidate.median_sharpe >= 0.95 * baseline.median_sharpe
    # 1.10x baseline — but if baseline is 0 (no trades / no drawdown),
    # any positive candidate drawdown fails the multiplicative gate.
    # Allow equal in that degenerate case.
    if baseline.median_max_drawdown == 0:
        g3_passed = candidate.median_max_drawdown <= 0
    else:
        g3_passed = (
            candidate.median_max_drawdown <= 1.10 * baseline.median_max_drawdown
        )

    gates = {
        "gate_1_f1": {
            "spec": GATE_F1_NEW_GTE_BASELINE,
            "passed": bool(g1_passed),
            "baseline": baseline.median_f1,
            "candidate": candidate.median_f1,
        },
        "gate_2_sharpe": {
            "spec": GATE_SHARPE_NEW_GTE_BASELINE_X_095,
            "passed": bool(g2_passed),
            "baseline": baseline.median_sharpe,
            "candidate": candidate.median_sharpe,
            "threshold": 0.95 * baseline.median_sharpe,
        },
        "gate_3_max_drawdown": {
            "spec": GATE_MAXDD_NEW_LTE_BASELINE_X_110,
            "passed": bool(g3_passed),
            "baseline": baseline.median_max_drawdown,
            "candidate": candidate.median_max_drawdown,
            "threshold": 1.10 * baseline.median_max_drawdown,
        },
    }

    passed = bool(g1_passed and g2_passed and g3_passed)
    failed = [name for name, g in gates.items() if not g["passed"]]
    rationale = (
        "all walk-forward gates passed"
        if passed
        else f"failed: {', '.join(failed)}"
    )
    return {"passed": passed, "gates": gates, "rationale": rationale}


# ---------------------------------------------------------------------------
# Persistence — write the artifact specified in scope §5.
# ---------------------------------------------------------------------------


def write_walkforward_artifact(
    output_path: Path | str,
    *,
    baseline: WalkforwardResult,
    candidate: WalkforwardResult,
    decision: Optional[dict] = None,
) -> Path:
    """Persist the walk-forward output to JSON in the canonical shape.

    Filename convention per scope §5:
    ``models/phase7a_walkforward_<instrument>.json``. The artifact
    holds both feature-set results, the gate decision, and a metadata
    block so a CI pipeline can read it without rerunning the harness.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    decision = decision or acceptance_decision(baseline, candidate)
    payload = {
        "instrument": baseline.instrument,
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
        "decision": decision,
    }
    output.write_text(json.dumps(payload, indent=2, default=str))
    return output
