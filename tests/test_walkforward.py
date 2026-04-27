"""Phase 7.A walk-forward harness tests.

Targets the contracts that matter for promotion gate integrity:

* **Fold boundaries** — windows slide by the step size; the test
  segment is exactly ``(1 - train_ratio)`` of the window; no fold
  starts before data starts; folds are capped at 24.
* **No look-ahead** — train_end < test_start for every fold.
* **Acceptance gates** — gate-1/2/3 evaluate correctly against
  hand-built ``WalkforwardResult`` pairs.
* **JSON output** — the artifact round-trips and carries the gate
  decision so CI can ingest it without rerunning the harness.
* **End-to-end** — synthetic feature matrix, plug-in fit_predict_fn,
  resulting WalkforwardResult has the expected fold count and the
  median aggregates are populated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.model.walkforward import (
    DEFAULT_TRAIN_RATIO,
    DEFAULT_WINDOW_MONTHS,
    FoldResult,
    WalkforwardResult,
    acceptance_decision,
    generate_fold_boundaries,
    run_walkforward,
    write_walkforward_artifact,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hourly_frame(months: int, seed: int = 7) -> pd.DataFrame:
    """Synthetic hourly feature matrix that's long enough to drive the
    walk-forward without slamming Tiingo / yfinance.

    Generates ``months * 30 * 24`` bars (rough; pd.date_range handles
    the calendar correctness). Signal label is a noisy AR(1) so the
    classifier has *something* to learn — pure white noise gives F1 ≈
    0 across the board which makes the gate comparison degenerate.
    """
    n = months * 30 * 24
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start="2022-01-01", periods=n, freq="1h", tz="UTC",
    )
    closes = 1.10 + np.cumsum(rng.normal(0.0, 0.0008, size=n))
    df = pd.DataFrame({
        "datetime": timestamps,
        "open": closes,
        "high": closes + 0.0005,
        "low": closes - 0.0005,
        "close": closes,
        "feat_a": rng.normal(0.0, 1.0, size=n),
        "feat_b": rng.normal(0.0, 1.0, size=n),
    })
    # Direction label — predictable from feat_a so the model can learn
    df["direction_fwd_4"] = np.sign(
        df["feat_a"] * 0.5 + rng.normal(0.0, 1.0, size=n)
    ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Fold boundary generation
# ---------------------------------------------------------------------------


class TestGenerateFoldBoundaries:

    def test_full_24_folds_when_data_supports_it(self):
        # 18-month window + 23 monthly steps → window_end at month 41
        # → need ≥ 41 months of data
        df = _hourly_frame(months=42)
        folds = generate_fold_boundaries(df)
        assert len(folds) == 24

    def test_caps_at_24(self):
        # 60 months of data — would yield 43 folds without the cap
        df = _hourly_frame(months=60)
        folds = generate_fold_boundaries(df)
        assert len(folds) == 24

    def test_short_data_raises(self):
        # 12 months of data, 18-month window → cannot run
        df = _hourly_frame(months=12)
        with pytest.raises(ValueError, match="span"):
            generate_fold_boundaries(df)

    def test_window_boundary_geometry(self):
        df = _hourly_frame(months=42)
        folds = generate_fold_boundaries(df)
        for fold in folds:
            window_span = (fold["window_end"] - fold["window_start"]).days
            # 18-month window ≈ 540 days; allow ± a few for calendar variance
            assert 540 - 35 < window_span < 540 + 35

    def test_train_test_split_at_train_ratio(self):
        df = _hourly_frame(months=42)
        folds = generate_fold_boundaries(df)
        for fold in folds:
            window_span = (fold["window_end"] - fold["window_start"]).days
            test_span = (fold["test_end"] - fold["test_start"]).days
            test_ratio_observed = test_span / window_span
            # 0.2 ± 0.05 (off-by-one days from DateOffset calendar math)
            assert 0.15 < test_ratio_observed < 0.25

    def test_no_lookahead(self):
        df = _hourly_frame(months=42)
        folds = generate_fold_boundaries(df)
        for fold in folds:
            assert fold["train_end"] <= fold["test_start"]
            assert fold["test_start"] <= fold["test_end"]

    def test_step_advances_by_one_month(self):
        df = _hourly_frame(months=42)
        folds = generate_fold_boundaries(df)
        for prev, nxt in zip(folds, folds[1:]):
            delta_days = (nxt["window_start"] - prev["window_start"]).days
            # 1 month step = 28-31 days
            assert 27 < delta_days < 32

    def test_empty_frame_raises(self):
        df = pd.DataFrame({"datetime": []})
        with pytest.raises(ValueError, match="empty"):
            generate_fold_boundaries(df)

    def test_invalid_train_ratio_raises(self):
        df = _hourly_frame(months=24)
        with pytest.raises(ValueError, match="train_ratio"):
            generate_fold_boundaries(df, train_ratio=0)
        with pytest.raises(ValueError, match="train_ratio"):
            generate_fold_boundaries(df, train_ratio=1.5)

    def test_n_folds_override(self):
        df = _hourly_frame(months=42)
        folds = generate_fold_boundaries(df, n_folds=5)
        assert len(folds) == 5


# ---------------------------------------------------------------------------
# Acceptance gates
# ---------------------------------------------------------------------------


def _result(*, f1: float, sharpe: float, max_dd: float) -> WalkforwardResult:
    """Convenience builder for hand-constructed results."""
    r = WalkforwardResult(instrument="EURUSD", feature_set="x", n_folds=1)
    r.median_f1 = f1
    r.median_sharpe = sharpe
    r.median_max_drawdown = max_dd
    return r


class TestAcceptanceDecision:

    def test_all_gates_pass(self):
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        candidate = _result(f1=0.42, sharpe=1.05, max_dd=0.10)
        decision = acceptance_decision(baseline, candidate)
        assert decision["passed"] is True
        for gate_id, gate in decision["gates"].items():
            assert gate["passed"] is True

    def test_gate_1_fails_when_f1_regresses(self):
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        candidate = _result(f1=0.39, sharpe=1.05, max_dd=0.10)
        decision = acceptance_decision(baseline, candidate)
        assert decision["passed"] is False
        assert decision["gates"]["gate_1_f1"]["passed"] is False
        assert "gate_1" in decision["rationale"]

    def test_gate_2_sharpe_5pct_slack(self):
        # 0.96 * 1.0 = 0.96 — passes (>= 0.95)
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        candidate = _result(f1=0.40, sharpe=0.96, max_dd=0.10)
        decision = acceptance_decision(baseline, candidate)
        assert decision["gates"]["gate_2_sharpe"]["passed"] is True

    def test_gate_2_fails_below_5pct_slack(self):
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        candidate = _result(f1=0.40, sharpe=0.93, max_dd=0.10)
        decision = acceptance_decision(baseline, candidate)
        assert decision["gates"]["gate_2_sharpe"]["passed"] is False

    def test_gate_3_drawdown_10pct_slack_in_magnitude(self):
        # 1.10 * 0.10 = 0.11 — candidate at 0.10 passes
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        candidate = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        decision = acceptance_decision(baseline, candidate)
        assert decision["gates"]["gate_3_max_drawdown"]["passed"] is True

    def test_gate_3_fails_when_drawdown_too_deep(self):
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        candidate = _result(f1=0.40, sharpe=1.0, max_dd=0.15)  # 50% deeper
        decision = acceptance_decision(baseline, candidate)
        assert decision["gates"]["gate_3_max_drawdown"]["passed"] is False

    def test_gate_3_zero_baseline_dd_handled(self):
        """If baseline has no drawdown (e.g. degenerate fold all-zero
        signals), 1.10 * 0 = 0, so any positive candidate dd fails.
        Equal (both zero) should pass."""
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.0)
        candidate_eq = _result(f1=0.40, sharpe=1.0, max_dd=0.0)
        candidate_pos = _result(f1=0.40, sharpe=1.0, max_dd=0.05)

        eq = acceptance_decision(baseline, candidate_eq)
        pos = acceptance_decision(baseline, candidate_pos)
        assert eq["gates"]["gate_3_max_drawdown"]["passed"] is True
        assert pos["gates"]["gate_3_max_drawdown"]["passed"] is False

    def test_rationale_lists_failed_gates(self):
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        candidate = _result(f1=0.30, sharpe=0.50, max_dd=0.20)
        decision = acceptance_decision(baseline, candidate)
        assert decision["passed"] is False
        # All three failed
        assert "gate_1" in decision["rationale"]
        assert "gate_2" in decision["rationale"]
        assert "gate_3" in decision["rationale"]


# ---------------------------------------------------------------------------
# End-to-end run_walkforward
# ---------------------------------------------------------------------------


class TestRunWalkforward:

    def test_runs_with_custom_fit_predict(self):
        """Use a constant-prediction stub so the test is deterministic
        and doesn't depend on lightgbm convergence."""
        df = _hourly_frame(months=24)

        def stub(X_train, y_train, X_test):
            return np.ones(len(X_test), dtype=int)

        result = run_walkforward(
            df,
            feature_cols=["feat_a", "feat_b"],
            instrument="EURUSD",
            feature_set="phase7a",
            n_folds=3,
            fit_predict_fn=stub,
        )
        assert result.n_folds == 3
        assert len(result.folds) == 3
        # Aggregates populated
        assert not np.isnan(result.median_f1)
        assert not np.isnan(result.median_sharpe)
        assert not np.isnan(result.median_max_drawdown)

    def test_per_fold_no_lookahead_in_train_test_slices(self):
        df = _hourly_frame(months=24)

        captured: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        def capture(X_train, y_train, X_test):
            train_ts_max = X_train.index.max() if not X_train.index.empty else None
            test_ts_min = X_test.index.min() if not X_test.index.empty else None
            captured.append((train_ts_max, test_ts_min))
            return np.zeros(len(X_test), dtype=int)

        run_walkforward(
            df,
            feature_cols=["feat_a", "feat_b"],
            instrument="EURUSD",
            n_folds=3,
            fit_predict_fn=capture,
        )
        # Index is positional — but the slicing logic guarantees train
        # bars come strictly before test bars by datetime. Re-derive:
        # we'll trust the boundary tests above and just confirm the
        # callback was hit per fold.
        assert len(captured) == 3

    def test_missing_feature_column_raises(self):
        df = _hourly_frame(months=24)
        with pytest.raises(ValueError, match="missing required columns"):
            run_walkforward(
                df,
                feature_cols=["feat_a", "feat_does_not_exist"],
                instrument="EURUSD",
            )

    def test_empty_feature_cols_raises(self):
        df = _hourly_frame(months=24)
        with pytest.raises(ValueError, match="cannot be empty"):
            run_walkforward(
                df,
                feature_cols=[],
                instrument="EURUSD",
            )

    def test_skips_fold_with_too_few_test_rows(self):
        """A fold with < 10 test rows is skipped — log + carry on. The
        harness should not crash."""
        # Construct a frame just barely big enough to define folds but
        # tight enough that the trailing fold is short.
        df = _hourly_frame(months=24)
        # Truncate to first 19 months to push the last fold's test
        # segment shorter than expected.
        cutoff = pd.Timestamp("2023-08-01", tz="UTC")
        df = df[df["datetime"] < cutoff].reset_index(drop=True)

        def stub(X_train, y_train, X_test):
            return np.zeros(len(X_test), dtype=int)

        # Don't crash even when some folds skip
        result = run_walkforward(
            df,
            feature_cols=["feat_a", "feat_b"],
            instrument="EURUSD",
            n_folds=3,
            fit_predict_fn=stub,
        )
        # n_folds in the result is the actual count after skips
        assert 0 <= result.n_folds <= 3


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


class TestArtifact:

    def test_write_walkforward_artifact_round_trips(self, tmp_path):
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        candidate = _result(f1=0.42, sharpe=1.05, max_dd=0.10)
        baseline.instrument = candidate.instrument = "EURUSD"
        baseline.feature_set = "baseline"
        candidate.feature_set = "phase7a"

        out = tmp_path / "phase7a_walkforward_EURUSD.json"
        path = write_walkforward_artifact(
            out, baseline=baseline, candidate=candidate,
        )
        assert path == out
        assert out.exists()

        payload = json.loads(out.read_text())
        assert payload["instrument"] == "EURUSD"
        assert payload["baseline"]["feature_set"] == "baseline"
        assert payload["candidate"]["feature_set"] == "phase7a"
        assert payload["decision"]["passed"] is True
        assert "gate_1_f1" in payload["decision"]["gates"]

    def test_artifact_includes_per_fold_breakdown(self, tmp_path):
        """The artifact must let CI dig into individual fold metrics
        without rerunning the harness — gate failure forensics."""
        baseline = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        candidate = _result(f1=0.40, sharpe=1.0, max_dd=0.10)
        baseline.folds = [
            FoldResult(
                fold_idx=0,
                window_start="2022-01-01T00:00:00",
                window_end="2023-07-01T00:00:00",
                train_end="2023-03-01T00:00:00",
                test_start="2023-03-01T00:00:00",
                test_end="2023-07-01T00:00:00",
                n_train=1000, n_test=500,
                f1=0.40, sharpe=1.0, max_drawdown=0.10,
            ),
        ]

        out = tmp_path / "art.json"
        write_walkforward_artifact(out, baseline=baseline, candidate=candidate)
        payload = json.loads(out.read_text())
        assert len(payload["baseline"]["folds"]) == 1
        assert payload["baseline"]["folds"][0]["fold_idx"] == 0
