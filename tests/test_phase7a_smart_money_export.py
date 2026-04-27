"""Phase 7.A Track B — smart_money_export tests.

Covers:
  * JSONL reader (single file + windowed)
  * forward-compat: malformed/missing-field rows skipped, not raised
  * date-range and instrument filtering
  * parquet round-trip preserves nested dicts
  * summary stats over realistic audit records
  * gate-5 drift machinery: int_ordinal, bool, categorical, continuous
  * Bonferroni correction + ≤1/N pass rule from PHASE_7A_SCOPE.md §6.1
  * CLI: summary, export, diff (incl. exit code on pass/fail)
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.audit.schema import ALL_FIELDS, AUDIT_FIELDS, TIMESTAMP_FIELD
from prism.audit.smart_money_export import (
    FEATURE_TYPES,
    _parse_feature_specs,
    compare_feature,
    compare_features,
    main,
    read_audit_window,
    summarize,
    to_parquet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _record(
    instrument: str = "EURUSD",
    direction: str = "LONG",
    *,
    confidence: float = 0.72,
    audit_ts: str = "2026-04-20T08:00:00+00:00",
    signal_id: str = "test-id",
    htf_bias: dict | None = None,
    smart_money: dict | None = None,
) -> dict:
    return {
        "audit_ts": audit_ts,
        "instrument": instrument,
        "direction": direction,
        "confidence": confidence,
        "confidence_level": "MEDIUM",
        "signal_id": signal_id,
        "signal_time": audit_ts,
        "model_version": "prism_v2.0",
        "regime": "RISK_ON",
        "news_bias": "NEUTRAL",
        "htf_bias": htf_bias,
        "smart_money": smart_money,
    }


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _seed_state(tmp_path: Path) -> Path:
    """Lay down 3 days × 2 instruments of audit records in tmp."""
    base = tmp_path / "signal_audit"
    _write_jsonl(
        base / "EURUSD" / "2026-04-20.jsonl",
        [
            _record("EURUSD", "LONG",  confidence=0.65, audit_ts="2026-04-20T08:00:00+00:00", signal_id="e1"),
            _record("EURUSD", "SHORT", confidence=0.71, audit_ts="2026-04-20T16:00:00+00:00", signal_id="e2"),
        ],
    )
    _write_jsonl(
        base / "EURUSD" / "2026-04-21.jsonl",
        [_record("EURUSD", "LONG", confidence=0.83, audit_ts="2026-04-21T08:00:00+00:00", signal_id="e3")],
    )
    _write_jsonl(
        base / "XAUUSD" / "2026-04-21.jsonl",
        [
            _record(
                "XAUUSD", "LONG", confidence=0.78,
                audit_ts="2026-04-21T12:00:00+00:00", signal_id="x1",
                smart_money={
                    "ob": {"distance_pips": 12.0, "direction": "LONG"},
                    "sweep": None,
                    "po3": {"phase": "MANIPULATION", "is_entry_phase": True},
                },
            ),
        ],
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class TestReader:

    def test_window_loads_all_files_when_unfiltered(self, tmp_path):
        _seed_state(tmp_path)
        df = read_audit_window(tmp_path)
        assert len(df) == 4
        assert set(df["instrument"]) == {"EURUSD", "XAUUSD"}

    def test_window_filters_by_instrument(self, tmp_path):
        _seed_state(tmp_path)
        df = read_audit_window(tmp_path, instrument="EURUSD")
        assert len(df) == 3
        assert set(df["instrument"]) == {"EURUSD"}

    def test_window_filters_by_date_range(self, tmp_path):
        _seed_state(tmp_path)
        df = read_audit_window(
            tmp_path,
            start_date=date(2026, 4, 21),
            end_date=date(2026, 4, 21),
        )
        assert len(df) == 2  # one EURUSD + one XAUUSD on 4-21
        assert (df[TIMESTAMP_FIELD].dt.date == date(2026, 4, 21)).all()

    def test_window_returns_canonical_columns_when_empty(self, tmp_path):
        df = read_audit_window(tmp_path)
        assert list(df.columns) == list(ALL_FIELDS)
        assert df.empty

    def test_window_skips_malformed_filenames(self, tmp_path, caplog):
        base = tmp_path / "signal_audit" / "EURUSD"
        base.mkdir(parents=True)
        (base / "not-a-date.jsonl").write_text("")
        _write_jsonl(base / "2026-04-20.jsonl", [_record()])

        with caplog.at_level("WARNING"):
            df = read_audit_window(tmp_path)
        assert len(df) == 1
        assert any("Unexpected audit filename" in rec.message for rec in caplog.records)

    def test_jsonl_reader_skips_malformed_lines(self, tmp_path, caplog):
        path = tmp_path / "signal_audit" / "EURUSD" / "2026-04-20.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(_record(signal_id="ok-1")) + "\n"
            + "not valid json\n"                          # malformed
            + json.dumps({"audit_ts": "2026-04-20T08:00:00+00:00",
                          "instrument": "EURUSD"}) + "\n"  # missing fields
            + json.dumps(_record(signal_id="ok-2")) + "\n"
        )

        with caplog.at_level("WARNING"):
            df = read_audit_window(tmp_path)
        assert len(df) == 2
        assert set(df["signal_id"]) == {"ok-1", "ok-2"}
        # Two warnings: one JSONDecodeError, one schema-missing-fields
        assert sum("Skipping malformed" in r.message for r in caplog.records) == 2

    def test_jsonl_reader_tolerates_unknown_field_lenient_mode(self, tmp_path):
        """Forward-compat: writer ships a new field before reader pins."""
        path = tmp_path / "signal_audit" / "EURUSD" / "2026-04-20.jsonl"
        path.parent.mkdir(parents=True)
        rec = _record()
        rec["future_field_v2"] = "hello"
        path.write_text(json.dumps(rec) + "\n")

        df = read_audit_window(tmp_path)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------

class TestParquet:

    def test_round_trip_preserves_nested_dicts(self, tmp_path):
        _seed_state(tmp_path)
        df = read_audit_window(tmp_path)

        out = tmp_path / "out" / "audit.parquet"
        path = to_parquet(df, out)
        assert path == out
        assert out.exists()

        df2 = pd.read_parquet(out)
        assert len(df2) == len(df)
        assert list(df2.columns) == list(df.columns)

        xau = df2[df2["instrument"] == "XAUUSD"].iloc[0]
        # Nested dict survived the round trip
        assert xau["smart_money"]["ob"]["distance_pips"] == 12.0
        assert xau["smart_money"]["po3"]["phase"] == "MANIPULATION"

    def test_to_parquet_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame([_record()])
        path = to_parquet(df, tmp_path / "deep" / "nested" / "audit.parquet")
        assert path.exists()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummarize:

    def test_empty_dataframe_returns_zero_stub(self):
        df = pd.DataFrame(columns=list(ALL_FIELDS))
        result = summarize(df)
        assert result["total_signals"] == 0
        assert result["by_instrument"] == {}
        assert result["confidence"] is None
        assert result["smart_money_presence"] == {"ob": 0.0, "sweep": 0.0, "po3": 0.0}

    def test_counts_and_quantiles(self, tmp_path):
        _seed_state(tmp_path)
        df = read_audit_window(tmp_path)
        result = summarize(df)

        assert result["total_signals"] == 4
        assert result["by_instrument"] == {"EURUSD": 3, "XAUUSD": 1}
        assert result["by_direction"] == {"LONG": 3, "SHORT": 1}
        # XAUUSD has ob + po3 populated, others have None — 1/4 each
        assert result["smart_money_presence"]["ob"] == 0.25
        assert result["smart_money_presence"]["po3"] == 0.25
        assert result["smart_money_presence"]["sweep"] == 0.0
        # Confidence range pulled from the 4 seeded values
        assert result["confidence"]["min"] == pytest.approx(0.65)
        assert result["confidence"]["max"] == pytest.approx(0.83)


# ---------------------------------------------------------------------------
# Drift comparison machinery
# ---------------------------------------------------------------------------

class TestCompareFeature:

    def test_int_ordinal_same_distribution_does_not_reject(self):
        rng = np.random.default_rng(seed=42)
        live = pd.Series(rng.choice([0, 1, 2, 3], size=500, p=[0.1, 0.2, 0.3, 0.4]))
        hist = pd.Series(rng.choice([0, 1, 2, 3], size=2000, p=[0.1, 0.2, 0.3, 0.4]))

        result = compare_feature(live, hist, feature_type="int_ordinal")
        assert result["test"] == "chi_squared_gof"
        assert result["reject"] is False
        assert result["p_value"] > 0.01

    def test_int_ordinal_different_distribution_rejects(self):
        rng = np.random.default_rng(seed=42)
        live = pd.Series(rng.choice([0, 1, 2, 3], size=500, p=[0.7, 0.1, 0.1, 0.1]))
        hist = pd.Series(rng.choice([0, 1, 2, 3], size=2000, p=[0.1, 0.2, 0.3, 0.4]))

        result = compare_feature(live, hist, feature_type="int_ordinal")
        assert result["reject"] is True
        assert result["p_value"] < 0.01

    def test_bool_same_proportion_does_not_reject(self):
        rng = np.random.default_rng(seed=42)
        live = pd.Series(rng.random(500) < 0.3)
        hist = pd.Series(rng.random(2000) < 0.3)

        result = compare_feature(live, hist, feature_type="bool")
        assert result["test"] == "fishers_exact"
        assert result["reject"] is False

    def test_bool_different_proportion_rejects(self):
        rng = np.random.default_rng(seed=42)
        live = pd.Series(rng.random(500) < 0.7)
        hist = pd.Series(rng.random(2000) < 0.3)

        result = compare_feature(live, hist, feature_type="bool")
        assert result["reject"] is True

    def test_continuous_same_distribution_does_not_reject(self):
        rng = np.random.default_rng(seed=42)
        live = pd.Series(rng.normal(loc=10, scale=2, size=500))
        hist = pd.Series(rng.normal(loc=10, scale=2, size=2000))

        result = compare_feature(live, hist, feature_type="continuous")
        assert result["test"] == "ks_two_sample"
        assert result["reject"] is False

    def test_continuous_shifted_distribution_rejects(self):
        rng = np.random.default_rng(seed=42)
        live = pd.Series(rng.normal(loc=15, scale=2, size=500))
        hist = pd.Series(rng.normal(loc=10, scale=2, size=2000))

        result = compare_feature(live, hist, feature_type="continuous")
        assert result["reject"] is True

    def test_categorical_dispatches_to_chi_squared(self):
        rng = np.random.default_rng(seed=42)
        cats = ["accumulation", "manipulation", "distribution", "unknown"]
        live = pd.Series(rng.choice(cats, size=400))
        hist = pd.Series(rng.choice(cats, size=2000))

        result = compare_feature(live, hist, feature_type="categorical")
        assert result["test"] == "chi_squared_gof"

    def test_unknown_feature_type_raises(self):
        with pytest.raises(ValueError, match="unknown feature_type"):
            compare_feature(
                pd.Series([1, 2]), pd.Series([1, 2]),
                feature_type="bogus",
            )

    def test_empty_sample_raises(self):
        with pytest.raises(ValueError):
            compare_feature(
                pd.Series([], dtype=float), pd.Series([1.0, 2.0, 3.0]),
                feature_type="continuous",
            )

    def test_alpha_threshold_respected(self):
        """Reject decision flips at the alpha = p_value boundary."""
        rng = np.random.default_rng(seed=99)
        live = pd.Series(rng.normal(loc=10.3, scale=2, size=200))
        hist = pd.Series(rng.normal(loc=10.0, scale=2, size=200))

        baseline = compare_feature(live, hist, feature_type="continuous", alpha=0.5)
        p = baseline["p_value"]

        # alpha > p ⇒ reject (more permissive threshold)
        permissive = compare_feature(
            live, hist, feature_type="continuous", alpha=p * 2,
        )
        assert permissive["reject"] is True

        # alpha < p ⇒ don't reject (more stringent threshold)
        stringent = compare_feature(
            live, hist, feature_type="continuous", alpha=p / 2,
        )
        assert stringent["reject"] is False


class TestCompareFeatures:
    """Multi-feature gate logic — Bonferroni + ≤1/N pass rule."""

    def _matched_dfs(self, n_live=400, n_hist=2000, seed=42):
        rng = np.random.default_rng(seed=seed)
        return (
            pd.DataFrame({
                "htf_alignment": rng.choice([0, 1, 2, 3], size=n_live, p=[0.1, 0.2, 0.3, 0.4]),
                "kill_zone_strength": rng.choice([0, 1, 2, 3], size=n_live),
                "sweep_confirmed": rng.random(n_live) < 0.3,
                "ob_distance_pips": rng.normal(loc=15, scale=8, size=n_live),
                "po3_phase": rng.choice(["acc", "man", "dist", "unk"], size=n_live),
            }),
            pd.DataFrame({
                "htf_alignment": rng.choice([0, 1, 2, 3], size=n_hist, p=[0.1, 0.2, 0.3, 0.4]),
                "kill_zone_strength": rng.choice([0, 1, 2, 3], size=n_hist),
                "sweep_confirmed": rng.random(n_hist) < 0.3,
                "ob_distance_pips": rng.normal(loc=15, scale=8, size=n_hist),
                "po3_phase": rng.choice(["acc", "man", "dist", "unk"], size=n_hist),
            }),
        )

    def _specs(self):
        return [
            ("htf_alignment", "int_ordinal"),
            ("kill_zone_strength", "int_ordinal"),
            ("sweep_confirmed", "bool"),
            ("ob_distance_pips", "continuous"),
            ("po3_phase", "categorical"),
        ]

    def test_bonferroni_alpha_is_family_over_n(self):
        live, hist = self._matched_dfs()
        result = compare_features(live, hist, self._specs(), family_alpha=0.05)
        assert result["per_feature_alpha"] == pytest.approx(0.01)
        assert result["n_features"] == 5

    def test_matched_distributions_pass(self):
        live, hist = self._matched_dfs()
        result = compare_features(live, hist, self._specs())
        assert result["passed"] is True
        assert result["rejections"] <= 1

    def test_two_features_drifted_fails(self):
        live, hist = self._matched_dfs()
        # Drift two features hard
        live["htf_alignment"] = 0
        live["sweep_confirmed"] = True

        result = compare_features(live, hist, self._specs())
        assert result["passed"] is False
        assert result["rejections"] >= 2

    def test_single_drift_still_passes_with_default_threshold(self):
        live, hist = self._matched_dfs()
        live["htf_alignment"] = 0  # only one feature drifted

        result = compare_features(live, hist, self._specs())
        assert result["rejections"] == 1
        assert result["passed"] is True

    def test_missing_column_raises(self):
        live, hist = self._matched_dfs()
        live = live.drop(columns=["htf_alignment"])
        with pytest.raises(KeyError, match="live_df missing column"):
            compare_features(live, hist, self._specs())

    def test_empty_specs_raises(self):
        live, hist = self._matched_dfs()
        with pytest.raises(ValueError, match="non-empty"):
            compare_features(live, hist, [])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:

    def test_summary_subcommand_prints_json(self, tmp_path, capsys):
        _seed_state(tmp_path)
        rc = main(["summary", "--state-dir", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        result = json.loads(out)
        assert result["total_signals"] == 4

    def test_summary_on_empty_state_dir(self, tmp_path, capsys):
        rc = main(["summary", "--state-dir", str(tmp_path)])
        assert rc == 0
        result = json.loads(capsys.readouterr().out)
        assert result["total_signals"] == 0

    def test_export_writes_parquet(self, tmp_path, capsys):
        _seed_state(tmp_path)
        out = tmp_path / "export.parquet"
        rc = main([
            "export", "--state-dir", str(tmp_path),
            "--output", str(out),
        ])
        assert rc == 0
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 4

    def test_diff_subcommand_pass_returns_zero(self, tmp_path, capsys):
        rng = np.random.default_rng(seed=42)
        df = pd.DataFrame({
            "x": rng.normal(loc=0, scale=1, size=2000),
        })
        live_path = tmp_path / "live.parquet"
        hist_path = tmp_path / "hist.parquet"
        df.iloc[:500].to_parquet(live_path)
        df.iloc[500:].to_parquet(hist_path)

        rc = main([
            "diff",
            "--live", str(live_path),
            "--historical", str(hist_path),
            "--features", "x:continuous",
        ])
        assert rc == 0  # passed
        result = json.loads(capsys.readouterr().out)
        assert result["passed"] is True

    def test_diff_subcommand_fail_returns_nonzero(self, tmp_path, capsys):
        rng = np.random.default_rng(seed=42)
        live_df = pd.DataFrame({
            "a": rng.normal(loc=5, scale=1, size=500),
            "b": rng.normal(loc=5, scale=1, size=500),
        })
        hist_df = pd.DataFrame({
            "a": rng.normal(loc=0, scale=1, size=2000),
            "b": rng.normal(loc=0, scale=1, size=2000),
        })
        live_path = tmp_path / "live.parquet"
        hist_path = tmp_path / "hist.parquet"
        live_df.to_parquet(live_path)
        hist_df.to_parquet(hist_path)

        rc = main([
            "diff",
            "--live", str(live_path),
            "--historical", str(hist_path),
            "--features", "a:continuous,b:continuous",
            "--max-rejections", "0",
        ])
        assert rc == 1
        result = json.loads(capsys.readouterr().out)
        assert result["passed"] is False

    def test_feature_spec_parser_validates_type(self):
        with pytest.raises(Exception, match="unknown feature type"):
            _parse_feature_specs("htf_alignment:bogus")

    def test_feature_spec_parser_validates_format(self):
        with pytest.raises(Exception, match="expected col:type"):
            _parse_feature_specs("htf_alignment")

    def test_module_invocation_smoke(self, tmp_path):
        """python -m prism.audit.smart_money_export summary --state-dir <empty>
        should exit zero and emit valid JSON."""
        result = subprocess.run(
            [
                sys.executable, "-m", "prism.audit.smart_money_export",
                "summary", "--state-dir", str(tmp_path),
            ],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(result.stdout)
        assert payload["total_signals"] == 0
