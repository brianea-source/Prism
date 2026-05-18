"""Tests for the Dukascopy fetch + Stockraft fetch pipeline.

Covers the two regression-prone areas in scripts/fetch_backtest_data.py and
prism/data/dukascopy.py:

1. bi5 record decoding (OCLH field order, seconds-from-midnight time base).
2. Resample agg API (must use the pandas 2.x tuple form).
3. Parquet schema produced by fetch_dukascopy.

Mocks Dukascopy HTTP and exercises the real lzma / bi5 parsing path.
"""
from __future__ import annotations

import io
import lzma
import struct
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from prism.data import dukascopy as duk


# ---------------------------------------------------------------------------
# Synthetic bi5 generator
# ---------------------------------------------------------------------------

def _synthesize_bi5(n_records: int = 1440, start_price: float = 1.10000,
                    point_factor: float = 100000.0) -> bytes:
    """Build a synthetic Dukascopy M1-candle bi5 file.

    Field order: (seconds_from_midnight, open, close, low, high, volume_f32).
    Encodes a slow drift upward to keep the chain
        close[i] == open[i+1]
    invariant intact, mirroring the real feed.
    """
    buf = io.BytesIO()
    p = start_price
    for i in range(n_records):
        secs = i * 60
        o = p
        # tiny random-ish drift; deterministic for tests
        delta = ((i * 17) % 13 - 6) * 1e-5     # \u00b1 6 pips
        c = o + delta
        h = max(o, c) + 1e-5
        l = min(o, c) - 1e-5
        vol = 0.1 + (i % 7) * 0.01
        rec = struct.pack(
            ">IIIIIf",
            secs,
            int(round(o * point_factor)),
            int(round(c * point_factor)),
            int(round(l * point_factor)),
            int(round(h * point_factor)),
            float(vol),
        )
        buf.write(rec)
        p = c
    return lzma.compress(buf.getvalue(), preset=6)


# ---------------------------------------------------------------------------
# _parse_bi5
# ---------------------------------------------------------------------------

class TestParseBi5:
    def test_record_count_matches_minute_count(self):
        raw = _synthesize_bi5(n_records=1440)
        df = duk._parse_bi5(raw, datetime(2025, 6, 16, tzinfo=timezone.utc), 100000.0)
        assert len(df) == 1440

    def test_time_base_is_seconds_not_milliseconds(self):
        raw = _synthesize_bi5(n_records=10)
        df = duk._parse_bi5(raw, datetime(2025, 6, 16, tzinfo=timezone.utc), 100000.0)
        # Consecutive bars must be 60s apart, not 60ms.
        diffs = df["datetime"].diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == pd.Timedelta(seconds=60)

    def test_ohlc_invariants_hold(self):
        raw = _synthesize_bi5(n_records=500)
        df = duk._parse_bi5(raw, datetime(2025, 6, 16, tzinfo=timezone.utc), 100000.0)
        # low <= open,close <= high, on every row.
        assert (df["low"] <= df["open"]).all()
        assert (df["open"] <= df["high"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["close"] <= df["high"]).all()

    def test_chain_invariant_close_equals_next_open(self):
        # Dukascopy's M1 stream is continuous: close[i] == open[i+1].
        raw = _synthesize_bi5(n_records=200)
        df = duk._parse_bi5(raw, datetime(2025, 6, 16, tzinfo=timezone.utc), 100000.0)
        # Within float rounding of point_factor.
        diff = (df["close"].iloc[:-1].values - df["open"].iloc[1:].values)
        assert (abs(diff) < 1e-4).all()

    def test_empty_raw_returns_empty_frame(self):
        # An lzma-empty payload shouldn't blow up.
        empty = lzma.compress(b"")
        df = duk._parse_bi5(empty, datetime(2025, 6, 16, tzinfo=timezone.utc), 100000.0)
        assert df.empty


# ---------------------------------------------------------------------------
# _resample_df (pandas 2.x tuple-agg API)
# ---------------------------------------------------------------------------

class TestResample:
    def _m1_frame(self, n=180):
        raw = _synthesize_bi5(n_records=n)
        return duk._parse_bi5(raw, datetime(2025, 6, 16, tzinfo=timezone.utc), 100000.0)

    def test_resample_h1_produces_correct_bar_count(self):
        df = self._m1_frame(n=180)        # 3 hours of M1 bars
        out = duk._resample_df(df, "H1")
        assert len(out) == 3
        # And the columns are the canonical OHLCV set.
        assert set(out.columns) >= {"datetime", "open", "high", "low", "close", "volume"}

    def test_resample_preserves_ohlc_semantics(self):
        df = self._m1_frame(n=60)
        out = duk._resample_df(df, "H1")
        assert len(out) == 1
        row = out.iloc[0]
        assert row["open"] == df["open"].iloc[0]
        assert row["close"] == df["close"].iloc[-1]
        assert row["high"] == df["high"].max()
        assert row["low"] == df["low"].min()
        assert row["volume"] == pytest.approx(df["volume"].sum())


# ---------------------------------------------------------------------------
# fetch_dukascopy end-to-end (HTTP mocked)
# ---------------------------------------------------------------------------

class TestFetchDukascopy:
    def test_schema_and_source_column(self, tmp_path, monkeypatch):
        # Force the module's cache dir into tmp so we don't touch real data/.
        monkeypatch.setattr(duk, "CACHE_DIR", tmp_path / "duka_cache")

        synthetic = _synthesize_bi5(n_records=1440)

        def fake_fetch(instrument, dt):
            return synthetic

        monkeypatch.setattr(duk, "_fetch_day_raw", fake_fetch)

        df = duk.fetch_dukascopy("EURUSD", "H1", "2025-06-16", "2025-06-17")
        assert df is not None and not df.empty

        # Schema
        expected_cols = {"datetime", "open", "high", "low", "close", "volume", "source"}
        assert expected_cols <= set(df.columns)
        assert (df["source"] == "dukascopy").all()

        # 24 H1 bars from one day of M1 data.
        assert len(df) == 24

        # Datetime is UTC tz-aware.
        assert str(df["datetime"].dt.tz) == "UTC"

        # OHLC validity preserved through resample.
        assert (df["low"] <= df["open"]).all()
        assert (df["open"] <= df["high"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["close"] <= df["high"]).all()

    def test_weekend_days_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(duk, "CACHE_DIR", tmp_path / "duka_cache")
        calls = []

        def fake_fetch(instrument, dt):
            calls.append(dt.date())
            return _synthesize_bi5(n_records=60)

        monkeypatch.setattr(duk, "_fetch_day_raw", fake_fetch)

        # 2025-06-14 = Sat, 2025-06-15 = Sun \u2014 must be skipped.
        duk.fetch_dukascopy("EURUSD", "H1", "2025-06-13", "2025-06-17")
        weekdays = [d for d in calls if d.weekday() < 5]
        weekends = [d for d in calls if d.weekday() >= 5]
        assert len(weekends) == 0
        assert len(weekdays) >= 1
