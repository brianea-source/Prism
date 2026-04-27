"""Phase 7.A historical state builder tests.

Covers:

* Input validation — required OHLC columns enforced for all three frames
* Warmup — entry frames shorter than warmup return an empty schema-shaped DF
* Schema — every emitted row matches the canonical audit-log column set
* Detector wiring — the smart-money sub-dicts have the same shape as the
  live runner persists (so feature_engineering.enrich_features works
  identically against historical and live data)
* Per-bar correctness — the builder emits the expected row count for a
  warmup-aware walk
* Parquet round-trip — write → read recovers the canonical schema
* signal_conditioned_only mode — when a detector mocks itself into
  failing, conditioned mode drops the row, default mode keeps it
* CLI — module-level invocation sanity
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from prism.audit.schema import ALL_FIELDS, TIMESTAMP_FIELD
from prism.data.historical_state import (
    DEFAULT_WARMUP_BARS,
    HistoricalStateBuilder,
    build_replay_sidecar,
    main,
    read_replay_sidecar,
    read_sidecar_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_ohlc(
    n: int,
    start: str = "2026-01-01T00:00:00+00:00",
    freq: str = "5min",
    seed: int = 42,
    drift: float = 0.0001,
) -> pd.DataFrame:
    """Generate a deterministic OHLC frame that looks like a real bar series.

    Uses a random walk so swing structure can form (the HTF bias engine
    needs alternating highs/lows; a flat or monotone series gives every
    timeframe ``RANGING`` and bypasses interesting paths).
    """
    rng = np.random.default_rng(seed=seed)
    timestamps = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    closes = 1.10 + np.cumsum(rng.normal(loc=drift, scale=0.0008, size=n))
    highs = closes + rng.uniform(0.0001, 0.0005, size=n)
    lows = closes - rng.uniform(0.0001, 0.0005, size=n)
    opens = closes + rng.uniform(-0.0003, 0.0003, size=n)
    return pd.DataFrame({
        "datetime": timestamps,
        "open": opens,
        "high": np.maximum.reduce([opens, highs, closes]),
        "low": np.minimum.reduce([opens, lows, closes]),
        "close": closes,
    })


def _frames(
    n_entry: int = 50, n_h1: int = 20, n_h4: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three timeframe frames sharing a UTC origin so windowing aligns."""
    df_h4 = _synthetic_ohlc(n_h4, freq="4h", seed=4)
    df_h1 = _synthetic_ohlc(n_h1, freq="1h", seed=1)
    df_entry = _synthetic_ohlc(n_entry, freq="5min", seed=5)
    return df_h4, df_h1, df_entry


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:

    def test_missing_columns_raises(self):
        df_h4, df_h1, _ = _frames()
        bad = pd.DataFrame({"datetime": [pd.Timestamp.now(tz="UTC")]})
        with pytest.raises(ValueError, match="missing required columns"):
            HistoricalStateBuilder(
                "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=bad,
            )

    def test_warmup_bar_floor_one(self):
        df_h4, df_h1, df_entry = _frames()
        b = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            warmup_bars=0,
        )
        assert b.warmup_bars == 1


# ---------------------------------------------------------------------------
# Build behaviour
# ---------------------------------------------------------------------------


class TestBuild:

    def test_short_entry_frame_returns_empty(self):
        df_h4, df_h1, _ = _frames()
        df_entry = _synthetic_ohlc(10, freq="5min", seed=5)
        b = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
        )
        result = b.build()
        assert result.empty
        assert list(result.columns) == list(ALL_FIELDS)

    def test_emits_one_row_per_bar_past_warmup(self):
        df_h4, df_h1, df_entry = _frames(n_entry=50)
        b = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            warmup_bars=21,
        )
        result = b.build()
        assert len(result) == 50 - 21

    def test_canonical_schema_columns_present(self):
        df_h4, df_h1, df_entry = _frames()
        b = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
        )
        result = b.build()
        for col in ALL_FIELDS:
            assert col in result.columns, f"missing canonical column: {col}"

    def test_audit_ts_is_iso_string(self):
        df_h4, df_h1, df_entry = _frames()
        b = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
        )
        result = b.build()
        first = result.iloc[0][TIMESTAMP_FIELD]
        # pd.Timestamp can parse it back round-trip
        ts = pd.Timestamp(first)
        assert ts.tz is not None  # tz-aware

    def test_smart_money_dict_shape_matches_live_runner(self):
        """The keys inside ``smart_money["ob"]`` / ``["sweep"]`` /
        ``["po3"]`` must match what ``generator._evaluate_smart_money``
        emits, because ``feature_engineering.enrich_features`` reads the
        same field names on both sides of gate 5."""
        df_h4, df_h1, df_entry = _frames(n_entry=50)
        b = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
        )
        result = b.build()

        # At least one row should have a non-None smart_money. With 50
        # bars + a noisy random walk Po3 generally fires (at minimum
        # ACCUMULATION). If none does, the test fails loudly rather
        # than silently passing on a degenerate fixture.
        sm_rows = result[result["smart_money"].notna()]
        assert not sm_rows.empty, (
            "no smart_money populated rows — fixture degenerate or "
            "detectors fully failing"
        )

        sample_sm = sm_rows.iloc[0]["smart_money"]
        assert set(sample_sm.keys()) == {"ob", "sweep", "po3"}

        # Po3 in particular must populate after warmup
        po3 = sample_sm.get("po3")
        if po3 is not None:
            assert {"phase", "session", "is_entry_phase"}.issubset(po3.keys())

    def test_instrument_is_propagated(self):
        df_h4, df_h1, df_entry = _frames()
        b = HistoricalStateBuilder(
            "GBPUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
        )
        result = b.build()
        assert (result["instrument"] == "GBPUSD").all()

    def test_signal_id_unique_per_row(self):
        df_h4, df_h1, df_entry = _frames()
        b = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
        )
        result = b.build()
        assert result["signal_id"].is_unique


# ---------------------------------------------------------------------------
# signal_conditioned_only mode
# ---------------------------------------------------------------------------


class TestSignalConditioning:
    """Conditioning drops bars where any detector raised. Without
    raising detectors, the conditioned and unconditioned outputs are
    equal — so we monkey-patch one detector to raise on every other call
    and verify the row count matches."""

    def test_default_mode_keeps_all_post_warmup_bars(self):
        df_h4, df_h1, df_entry = _frames(n_entry=40)
        b = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            warmup_bars=21,
        )
        result = b.build()
        assert len(result) == 40 - 21

    def test_conditioned_mode_drops_failing_bars(self):
        df_h4, df_h1, df_entry = _frames(n_entry=40)
        b = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            warmup_bars=21,
            signal_conditioned_only=True,
        )

        original_detect = b.po3_detector.detect_phase
        call_counter = {"n": 0}

        def flaky(df, **kwargs):
            call_counter["n"] += 1
            if call_counter["n"] % 2 == 0:
                raise RuntimeError("simulated po3 failure")
            return original_detect(df, **kwargs)

        b.po3_detector.detect_phase = flaky  # type: ignore[assignment]

        result = b.build()
        # Half the post-warmup bars should be dropped
        assert len(result) < 40 - 21
        assert len(result) >= (40 - 21) // 2 - 1

    def test_conditioned_mode_does_not_drop_when_detectors_clean(self):
        df_h4, df_h1, df_entry = _frames(n_entry=40)
        b1 = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            warmup_bars=21,
            signal_conditioned_only=False,
        )
        b2 = HistoricalStateBuilder(
            "EURUSD", df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            warmup_bars=21,
            signal_conditioned_only=True,
        )
        assert len(b1.build()) == len(b2.build())


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------


class TestSidecarIO:

    def test_build_replay_sidecar_writes_parquet(self, tmp_path):
        df_h4, df_h1, df_entry = _frames(n_entry=50)
        out = tmp_path / "deep" / "nested" / "sidecar.parquet"
        path = build_replay_sidecar(
            instrument="EURUSD",
            df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            output_path=out,
        )
        assert path == out
        assert out.exists()

    def test_read_replay_sidecar_round_trips_columns(self, tmp_path):
        df_h4, df_h1, df_entry = _frames(n_entry=50)
        out = tmp_path / "sidecar.parquet"
        build_replay_sidecar(
            instrument="EURUSD",
            df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            output_path=out,
        )
        loaded = read_replay_sidecar(out)
        assert list(loaded.columns)[: len(ALL_FIELDS)] == list(ALL_FIELDS)
        assert len(loaded) > 0

    def test_read_sidecar_rejects_missing_canonical_columns(self, tmp_path):
        df = pd.DataFrame({"datetime": [pd.Timestamp.now(tz="UTC")]})
        out = tmp_path / "broken.parquet"
        df.to_parquet(out, index=False)
        with pytest.raises(ValueError, match="missing canonical columns"):
            read_replay_sidecar(out)


# ---------------------------------------------------------------------------
# Cross-module integration with feature_engineering
# ---------------------------------------------------------------------------


class TestFeatureEngineeringIntegration:
    """The whole point of the audit-log-shaped output: it feeds straight
    into ``enrich_features`` without translation."""

    def test_enrich_features_consumes_sidecar_directly(self, tmp_path):
        from prism.data.feature_engineering import (
            PHASE_7A_FEATURE_COLUMNS,
            enrich_features,
        )

        df_h4, df_h1, df_entry = _frames(n_entry=50)
        out = tmp_path / "sidecar.parquet"
        build_replay_sidecar(
            instrument="EURUSD",
            df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            output_path=out,
        )
        sidecar = read_replay_sidecar(out)
        enriched = enrich_features(sidecar, ob_max_distance_pips=30.0)

        for col in PHASE_7A_FEATURE_COLUMNS:
            assert col in enriched.columns, f"enrich missed column: {col}"
        # Every row should have a derivation, no NaN where we promised
        # a value.
        assert enriched["po3_phase"].notna().all()
        assert enriched["sweep_confirmed"].notna().all()
        assert enriched["ob_distance_pips"].notna().all()
        # ob_distance_pips values should be either -1 (no OB) or > 0
        valid = (enriched["ob_distance_pips"] == -1) | (
            enriched["ob_distance_pips"] > 0
        )
        assert valid.all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:

    def test_main_writes_sidecar_and_prints_metadata(self, tmp_path, capsys):
        df_h4, df_h1, df_entry = _frames(n_entry=50)
        h4_path = tmp_path / "h4.parquet"
        h1_path = tmp_path / "h1.parquet"
        entry_path = tmp_path / "entry.parquet"
        out_path = tmp_path / "sidecar.parquet"
        df_h4.to_parquet(h4_path)
        df_h1.to_parquet(h1_path)
        df_entry.to_parquet(entry_path)

        rc = main([
            "--instrument", "EURUSD",
            "--h4", str(h4_path),
            "--h1", str(h1_path),
            "--entry", str(entry_path),
            "--output", str(out_path),
        ])
        assert rc == 0
        assert out_path.exists()
        meta = json.loads(capsys.readouterr().out)
        assert meta["instrument"] == "EURUSD"
        assert meta["output"] == str(out_path)

    def test_module_invocation_smoke(self, tmp_path):
        df_h4, df_h1, df_entry = _frames(n_entry=50)
        h4_path = tmp_path / "h4.parquet"
        h1_path = tmp_path / "h1.parquet"
        entry_path = tmp_path / "entry.parquet"
        out_path = tmp_path / "sidecar.parquet"
        df_h4.to_parquet(h4_path)
        df_h1.to_parquet(h1_path)
        df_entry.to_parquet(entry_path)

        result = subprocess.run(
            [
                sys.executable, "-m", "prism.data.historical_state",
                "--instrument", "EURUSD",
                "--h4", str(h4_path),
                "--h1", str(h1_path),
                "--entry", str(entry_path),
                "--output", str(out_path),
            ],
            capture_output=True, text=True, check=True,
        )
        meta = json.loads(result.stdout)
        assert meta["instrument"] == "EURUSD"
        assert out_path.exists()


# ---------------------------------------------------------------------------
# N2: Parquet sidecar metadata (PRISM_OB_MAX_DISTANCE_PIPS audit trail)
# ---------------------------------------------------------------------------


class TestSidecarMetadata:

    def test_metadata_contains_instrument_and_ob_distance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "42.5")
        df_h4, df_h1, df_entry = _frames(n_entry=50)
        out = tmp_path / "sidecar.parquet"
        build_replay_sidecar(
            instrument="XAUUSD",
            df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            output_path=out,
        )
        meta = read_sidecar_metadata(out)
        assert meta["prism.instrument"] == "XAUUSD"
        assert meta["prism.ob_max_distance_pips"] == "42.5"
        assert "prism.built_at" in meta

    def test_metadata_empty_string_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRISM_OB_MAX_DISTANCE_PIPS", raising=False)
        df_h4, df_h1, df_entry = _frames(n_entry=50)
        out = tmp_path / "sidecar.parquet"
        build_replay_sidecar(
            instrument="EURUSD",
            df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
            output_path=out,
        )
        meta = read_sidecar_metadata(out)
        assert meta["prism.ob_max_distance_pips"] == ""

    def test_read_sidecar_metadata_on_non_prism_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        out = tmp_path / "plain.parquet"
        df.to_parquet(out, index=False)
        meta = read_sidecar_metadata(out)
        assert meta == {}


# ---------------------------------------------------------------------------
# N3: SweepDetector.latest_scanned_bar public property
# ---------------------------------------------------------------------------


class TestSweepDetectorProperty:

    def test_latest_scanned_bar_property_matches_private(self):
        from prism.signal.sweeps import SweepDetector
        det = SweepDetector("EURUSD")
        assert det.latest_scanned_bar is None
        df_entry = _synthetic_ohlc(50, freq="5min", seed=7)
        det.detect(df_entry)
        assert det.latest_scanned_bar == 49
        assert det.latest_scanned_bar == det._latest_scanned_bar
