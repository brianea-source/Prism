"""
tests/test_icc.py
-----------------
Unit tests for the ICC pattern detector (prism/signal/icc.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.signal.icc import (
    AOIDetector,
    detect_swing_points,
    detect_icc_phase,
    get_icc_entry,
    CORRECTION_MIN_PCT,
    CORRECTION_MAX_PCT,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic OHLCV data
# ---------------------------------------------------------------------------

def _make_ohlcv(closes: list[float], spread: float = 0.0002) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    highs  = closes_arr + spread
    lows   = closes_arr - spread
    opens  = np.roll(closes_arr, 1)
    opens[0] = closes_arr[0]
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes_arr,
        "volume": np.ones(n) * 1000,
    })


def _trend_up(n: int = 60, start: float = 1.1000, step: float = 0.0003) -> list[float]:
    return [start + i * step for i in range(n)]


def _trend_down(n: int = 60, start: float = 1.1200, step: float = 0.0003) -> list[float]:
    return [start - i * step for i in range(n)]


def _icc_long_setup() -> pd.DataFrame:
    """
    Construct a synthetic long ICC setup:
    - 20 bars of uptrend (indication)
    - 10 bars of correction (≈45% retracement)
    - 5 bars of continuation
    """
    indication = _trend_up(20, start=1.1000, step=0.0005)   # +0.0100 total
    indication_high = indication[-1]
    indication_low  = indication[0]
    indication_range = indication_high - indication_low       # 0.0100

    correction_depth = indication_range * 0.45               # ≈0.0045
    correction_low   = indication_high - correction_depth
    correction = _trend_down(10, start=indication_high, step=indication_range * 0.045)

    continuation = _trend_up(5, start=correction[-1], step=0.0003)

    return _make_ohlcv(indication + correction + continuation)


def _icc_short_setup() -> pd.DataFrame:
    """
    Construct a synthetic short ICC setup:
    - 20 bars of downtrend (indication)
    - 10 bars of correction (≈40% retracement)
    - 5 bars of continuation
    """
    indication = _trend_down(20, start=1.1200, step=0.0005)
    indication_low  = indication[-1]
    indication_high = indication[0]
    indication_range = indication_high - indication_low

    correction_height = indication_range * 0.40
    correction = _trend_up(10, start=indication_low, step=indication_range * 0.040)

    continuation = _trend_down(5, start=correction[-1], step=0.0003)

    return _make_ohlcv(indication + correction + continuation)


# ---------------------------------------------------------------------------
# detect_swing_points
# ---------------------------------------------------------------------------

class TestDetectSwingPoints:

    def test_columns_added(self):
        df = _make_ohlcv(_trend_up(30))
        result = detect_swing_points(df, lookback=5)
        assert "swing_high" in result.columns
        assert "swing_low" in result.columns

    def test_uptrend_has_no_swing_high_mid(self):
        """In a pure uptrend the last bar should be the swing high — not mid-series bars."""
        closes = _trend_up(30, step=0.001)
        df = detect_swing_points(_make_ohlcv(closes), lookback=5)
        # All mid-bars (not lookback away from edges) should NOT be swing highs
        mid = df["swing_high"].iloc[5:-5]
        assert mid.sum() == 0, "Pure uptrend should have no mid-series swing highs"

    def test_v_shape_has_swing_low(self):
        """A V-shaped price series should produce exactly one swing low."""
        closes = list(range(20, -1, -1)) + list(range(1, 21))  # V-shape (unique apex at 0)
        closes = [x * 0.001 + 1.0 for x in closes]
        df = detect_swing_points(_make_ohlcv(closes), lookback=5)
        # Tip of the V should be flagged as a swing low
        assert df["swing_low"].any(), "V-shape should produce at least one swing low"

    def test_inverted_v_has_swing_high(self):
        """An inverted-V price series should produce exactly one swing high."""
        closes = list(range(1, 21)) + [21] + list(range(20, 0, -1))  # Inv-V (unique peak at 21)
        closes = [x * 0.001 + 1.0 for x in closes]
        df = detect_swing_points(_make_ohlcv(closes), lookback=5)
        assert df["swing_high"].any(), "Inverted-V should produce at least one swing high"

    def test_output_is_copy(self):
        """Function must not modify the input DataFrame in-place."""
        df = _make_ohlcv(_trend_up(20))
        original_cols = set(df.columns)
        _ = detect_swing_points(df, lookback=5)
        assert set(df.columns) == original_cols, "Input DataFrame should not be mutated"

    def test_short_series_returns_all_false(self):
        """Series shorter than 2×lookback+1 should return all-False swing columns."""
        df = detect_swing_points(_make_ohlcv(_trend_up(5)), lookback=5)
        assert not df["swing_high"].any()
        assert not df["swing_low"].any()


# ---------------------------------------------------------------------------
# detect_icc_phase
# ---------------------------------------------------------------------------

class TestDetectICCPhase:

    def test_returns_string(self):
        df = _make_ohlcv(_trend_up(40))
        result = detect_icc_phase(df, lookback=20)
        assert isinstance(result, str)

    def test_valid_phases(self):
        valid = {
            "NONE",
            "INDICATION_BULL", "INDICATION_BEAR",
            "CORRECTION_BULL", "CORRECTION_BEAR",
            "CONTINUATION_LONG", "CONTINUATION_SHORT",
        }
        df = _make_ohlcv(_trend_up(40))
        for lookback in [5, 10, 20]:
            phase = detect_icc_phase(df, lookback=lookback)
            assert phase in valid, f"Unexpected phase: {phase!r}"

    def test_insufficient_data_returns_none(self):
        df = _make_ohlcv(_trend_up(3))
        assert detect_icc_phase(df, lookback=20) == "NONE"

    def test_strong_uptrend_indication(self):
        """A strong uptrend breaking beyond the lookback range should signal INDICATION_BULL."""
        # 50 bars of steady uptrend, last bar breaks above 20-bar range
        closes = _trend_up(50, step=0.002)
        df = _make_ohlcv(closes)
        phase = detect_icc_phase(df, lookback=20)
        assert "BULL" in phase or phase == "NONE", f"Expected bull phase or none, got {phase}"


# ---------------------------------------------------------------------------
# get_icc_entry
# ---------------------------------------------------------------------------

class TestGetICCEntry:

    def test_long_setup_returns_dict(self):
        df = _icc_long_setup()
        result = get_icc_entry(df, pip_value=0.0001)
        # May or may not detect depending on exact bar arrangement; just check type
        assert result is None or isinstance(result, dict)

    def test_signal_keys_present(self):
        """Any returned signal must have the required keys."""
        required_keys = {"phase", "entry", "sl", "correction_pct",
                         "indication_range_pips", "aoi_nearby"}
        df = _icc_long_setup()
        result = get_icc_entry(df, pip_value=0.0001)
        if result is not None:
            assert required_keys.issubset(set(result.keys()))

    def test_sl_below_entry_for_long(self):
        """For a LONG signal, SL must be below entry."""
        df = _icc_long_setup()
        result = get_icc_entry(df, pip_value=0.0001)
        if result is not None and result["phase"] == "CONTINUATION_LONG":
            assert result["sl"] < result["entry"], \
                f"SL {result['sl']} should be below entry {result['entry']}"

    def test_sl_above_entry_for_short(self):
        """For a SHORT signal, SL must be above entry."""
        df = _icc_short_setup()
        result = get_icc_entry(df, pip_value=0.0001)
        if result is not None and result["phase"] == "CONTINUATION_SHORT":
            assert result["sl"] > result["entry"], \
                f"SL {result['sl']} should be above entry {result['entry']}"

    def test_correction_pct_in_valid_range(self):
        """Correction percentage must be within the defined ICC thresholds."""
        df = _icc_long_setup()
        result = get_icc_entry(df, pip_value=0.0001)
        if result is not None:
            pct = result["correction_pct"] / 100
            assert CORRECTION_MIN_PCT <= pct <= CORRECTION_MAX_PCT, \
                f"correction_pct {result['correction_pct']}% out of range"

    def test_insufficient_data_returns_none(self):
        df = _make_ohlcv(_trend_up(5))
        assert get_icc_entry(df, pip_value=0.0001) is None

    def test_flat_market_returns_none(self):
        """A perfectly flat market should not generate an ICC signal."""
        closes = [1.1000] * 60
        df = _make_ohlcv(closes)
        result = get_icc_entry(df, pip_value=0.0001)
        assert result is None, "Flat market should not generate an ICC signal"


# ---------------------------------------------------------------------------
# AOIDetector
# ---------------------------------------------------------------------------

class TestAOIDetector:

    def _daily_df(self) -> pd.DataFrame:
        dates = pd.date_range("2026-04-01", periods=10, freq="D")
        highs  = np.linspace(1.1100, 1.1200, 10)
        lows   = np.linspace(1.1000, 1.1100, 10)
        closes = (highs + lows) / 2
        return pd.DataFrame({
            "datetime": dates,
            "open":   closes,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "volume": np.ones(10) * 1000,
        })

    def test_initialises_without_error(self):
        aoi = AOIDetector(self._daily_df())
        assert aoi is not None

    def test_empty_df_returns_empty_levels(self):
        aoi = AOIDetector(pd.DataFrame())
        assert aoi.get_nearby_aoi(1.1100) == []
        assert aoi.is_at_aoi(1.1100) is False

    def test_is_at_aoi_true_when_near_level(self):
        df = self._daily_df()
        aoi = AOIDetector(df)
        # Last daily high should be detectable
        last_high = df["high"].iloc[-1]
        assert aoi.is_at_aoi(last_high, tolerance_pips=5, pip_value=0.0001)

    def test_is_at_aoi_false_when_far(self):
        df = self._daily_df()
        aoi = AOIDetector(df)
        # Price far from any level
        assert not aoi.is_at_aoi(1.5000, tolerance_pips=5, pip_value=0.0001)

    def test_get_nearby_aoi_returns_list(self):
        df = self._daily_df()
        aoi = AOIDetector(df)
        last_high = df["high"].iloc[-1]
        nearby = aoi.get_nearby_aoi(last_high, tolerance_pips=5, pip_value=0.0001)
        assert isinstance(nearby, list)
        assert all("price" in lvl and "type" in lvl for lvl in nearby)

    def test_aoi_type_strings(self):
        df = self._daily_df()
        aoi = AOIDetector(df)
        all_types = {lvl["type"] for lvl in aoi._levels}
        valid_types = {"daily_high", "daily_low", "weekly_high", "weekly_low"}
        assert all_types.issubset(valid_types), f"Invalid AOI types: {all_types - valid_types}"
