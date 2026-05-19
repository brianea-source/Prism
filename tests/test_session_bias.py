"""Tests for the Session-level Po3 Bias Engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timezone

from prism.signal.session_bias import (
    SessionBiasEngine,
    SessionPhase,
    AsianRange,
    SessionBias,
)


# ---------------------------------------------------------------------------
# Helpers — build M5 bar DataFrames with UTC-aware datetime index
# ---------------------------------------------------------------------------

def _m5_bars(rows: list[dict]) -> pd.DataFrame:
    """Build an M5 DataFrame from a list of OHLC dicts with ``dt`` key."""
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df.pop("dt"))
    return df


def _asian_bars(ref: date, n: int = 72, base_price: float = 3200.0,
                range_size: float = 20.0) -> list[dict]:
    """Generate ``n`` M5 bars in the Asian session (00:00–06:00 UTC).

    Price oscillates within [base_price, base_price + range_size].
    """
    bars = []
    half = range_size / 2
    mid = base_price + half
    for i in range(n):
        minute = i * 5
        hour = minute // 60
        mins = minute % 60
        dt = datetime(ref.year, ref.month, ref.day, hour, mins, tzinfo=timezone.utc)
        price = mid + half * np.sin(2 * np.pi * i / n)
        bars.append({
            "dt": dt,
            "open": round(price - 0.5, 2),
            "high": round(price + 1.0, 2),
            "low": round(price - 1.0, 2),
            "close": round(price + 0.3, 2),
        })
    return bars


def _london_bars(ref: date, start_hour: int = 7, n: int = 24,
                 base_price: float = 3210.0, trend: float = 0.5) -> list[dict]:
    """Generate ``n`` M5 bars starting at ``start_hour`` UTC."""
    bars = []
    price = base_price
    for i in range(n):
        total_min = start_hour * 60 + i * 5
        hour = total_min // 60
        mins = total_min % 60
        dt = datetime(ref.year, ref.month, ref.day, hour, mins, tzinfo=timezone.utc)
        o = round(price, 2)
        h = round(price + 2.0, 2)
        l = round(price - 1.0, 2)
        c = round(price + trend, 2)
        bars.append({"dt": dt, "open": o, "high": h, "low": l, "close": c})
        price += trend
    return bars


# ---------------------------------------------------------------------------
# Asian range loading
# ---------------------------------------------------------------------------

class TestAsianRangeLoading:
    def test_loads_from_m5_bars(self):
        ref = date(2026, 5, 19)
        bars = _asian_bars(ref, n=72, base_price=3200.0, range_size=20.0)
        df = _m5_bars(bars)
        engine = SessionBiasEngine("XAUUSD")
        ar = engine.load_asian_range(df, ref_date=ref)

        assert ar is not None
        assert ar.bar_count == 72
        assert ar.date == ref
        assert ar.high >= ar.low
        assert ar.range_pips > 0
        assert ar.midpoint == pytest.approx((ar.high + ar.low) / 2, abs=0.01)

    def test_returns_none_with_insufficient_bars(self, monkeypatch):
        monkeypatch.setenv("PRISM_MIN_ASIAN_BARS", "50")
        ref = date(2026, 5, 19)
        bars = _asian_bars(ref, n=6)  # only 6 bars
        df = _m5_bars(bars)
        engine = SessionBiasEngine("XAUUSD")
        ar = engine.load_asian_range(df, ref_date=ref)
        assert ar is None

    def test_resets_on_new_day(self):
        engine = SessionBiasEngine("XAUUSD")
        ref1 = date(2026, 5, 19)
        ref2 = date(2026, 5, 20)
        bars1 = _asian_bars(ref1, n=72)
        bars2 = _asian_bars(ref2, n=72, base_price=3300.0)

        engine.load_asian_range(_m5_bars(bars1), ref_date=ref1)
        assert engine.asian_range is not None
        old_high = engine.asian_range.high

        engine.load_asian_range(_m5_bars(bars2), ref_date=ref2)
        assert engine.asian_range is not None
        assert engine.asian_range.date == ref2
        assert engine.asian_range.high != old_high

    def test_handles_datetime_column_instead_of_index(self):
        ref = date(2026, 5, 19)
        bars = _asian_bars(ref, n=20)
        df = pd.DataFrame(bars)
        df.rename(columns={"dt": "datetime"}, inplace=True)
        engine = SessionBiasEngine("XAUUSD")
        ar = engine.load_asian_range(df, ref_date=ref)
        assert ar is not None
        assert ar.bar_count == 20


# ---------------------------------------------------------------------------
# Sweep detection
# ---------------------------------------------------------------------------

class TestSweepDetection:
    def _make_engine_with_range(self, ref: date, asian_high=3220.0, asian_low=3200.0):
        engine = SessionBiasEngine("XAUUSD")
        engine._today = ref
        engine._asian_range = AsianRange(
            high=asian_high,
            low=asian_low,
            midpoint=(asian_high + asian_low) / 2,
            range_pips=(asian_high - asian_low) / 0.01,
            bar_count=72,
            date=ref,
        )
        return engine

    def test_high_sweep_sets_short_direction(self):
        ref = date(2026, 5, 19)
        engine = self._make_engine_with_range(ref, asian_high=3220.0, asian_low=3200.0)

        # London bar that sweeps above 3220 then closes back inside
        bars = [{
            "dt": datetime(2026, 5, 19, 8, 0, tzinfo=timezone.utc),
            "open": 3218.0, "high": 3225.0, "low": 3215.0, "close": 3216.0,
        }]
        df = _m5_bars(bars)
        bias = engine.update(df)

        assert bias.sweep_confirmed is True
        assert bias.sweep_side == "HIGH"
        assert bias.direction == "SHORT"
        assert bias.phase == SessionPhase.MANIPULATION

    def test_low_sweep_sets_long_direction(self):
        ref = date(2026, 5, 19)
        engine = self._make_engine_with_range(ref, asian_high=3220.0, asian_low=3200.0)

        bars = [{
            "dt": datetime(2026, 5, 19, 8, 0, tzinfo=timezone.utc),
            "open": 3202.0, "high": 3205.0, "low": 3195.0, "close": 3203.0,
        }]
        df = _m5_bars(bars)
        bias = engine.update(df)

        assert bias.sweep_confirmed is True
        assert bias.sweep_side == "LOW"
        assert bias.direction == "LONG"

    def test_no_sweep_returns_accumulation(self):
        ref = date(2026, 5, 19)
        engine = self._make_engine_with_range(ref, asian_high=3220.0, asian_low=3200.0)

        bars = [{
            "dt": datetime(2026, 5, 19, 8, 0, tzinfo=timezone.utc),
            "open": 3210.0, "high": 3215.0, "low": 3205.0, "close": 3212.0,
        }]
        df = _m5_bars(bars)
        bias = engine.update(df)

        assert bias.sweep_confirmed is False
        assert bias.direction is None
        assert bias.phase == SessionPhase.ACCUMULATION

    def test_break_not_sweep_when_close_outside(self):
        """A breakout (close above Asian high) is NOT a sweep — it's a real break."""
        ref = date(2026, 5, 19)
        engine = self._make_engine_with_range(ref, asian_high=3220.0, asian_low=3200.0)

        bars = [{
            "dt": datetime(2026, 5, 19, 8, 0, tzinfo=timezone.utc),
            "open": 3218.0, "high": 3228.0, "low": 3217.0, "close": 3226.0,
        }]
        df = _m5_bars(bars)
        bias = engine.update(df)

        assert bias.sweep_confirmed is False
        assert bias.direction is None


# ---------------------------------------------------------------------------
# Displacement detection
# ---------------------------------------------------------------------------

class TestDisplacement:
    def test_displacement_after_high_sweep(self, monkeypatch):
        monkeypatch.setenv("PRISM_SESSION_DISPLACEMENT_PIPS", "10")
        ref = date(2026, 5, 19)

        engine = SessionBiasEngine("XAUUSD")
        engine._today = ref
        engine._asian_range = AsianRange(
            high=3220.0, low=3200.0, midpoint=3210.0,
            range_pips=2000.0, bar_count=72, date=ref,
        )

        bars = [
            # Sweep bar: wick above 3220, close back inside
            {"dt": datetime(2026, 5, 19, 8, 0, tzinfo=timezone.utc),
             "open": 3218.0, "high": 3225.0, "low": 3215.0, "close": 3216.0},
            # Displacement: price drops 15 pips (0.15) from sweep
            {"dt": datetime(2026, 5, 19, 8, 5, tzinfo=timezone.utc),
             "open": 3216.0, "high": 3217.0, "low": 3210.0, "close": 3211.0},
        ]
        df = _m5_bars(bars)
        bias = engine.update(df)

        assert bias.sweep_confirmed is True
        assert bias.displacement_confirmed is True
        assert bias.phase == SessionPhase.DISTRIBUTION
        assert bias.direction == "SHORT"
        assert bias.confidence_modifier == 1.0

    def test_no_displacement_if_insufficient_move(self, monkeypatch):
        monkeypatch.setenv("PRISM_SESSION_DISPLACEMENT_PIPS", "50")
        ref = date(2026, 5, 19)

        engine = SessionBiasEngine("XAUUSD")
        engine._today = ref
        engine._asian_range = AsianRange(
            high=3220.0, low=3200.0, midpoint=3210.0,
            range_pips=2000.0, bar_count=72, date=ref,
        )

        # Sweep bar: wick above 3220, close back inside.
        # Follow-up bar moves only 0.20 from sweep (20 pips), under 50 pip threshold.
        bars = [
            {"dt": datetime(2026, 5, 19, 8, 0, tzinfo=timezone.utc),
             "open": 3218.0, "high": 3220.30, "low": 3218.0, "close": 3219.0},
            {"dt": datetime(2026, 5, 19, 8, 5, tzinfo=timezone.utc),
             "open": 3219.0, "high": 3219.50, "low": 3220.0, "close": 3219.50},
        ]
        df = _m5_bars(bars)
        bias = engine.update(df)

        assert bias.sweep_confirmed is True
        assert bias.displacement_confirmed is False
        assert bias.phase == SessionPhase.MANIPULATION
        assert bias.confidence_modifier == 0.6


# ---------------------------------------------------------------------------
# No data edge case
# ---------------------------------------------------------------------------

class TestNoData:
    def test_update_without_asian_range_returns_no_data(self):
        engine = SessionBiasEngine("XAUUSD")
        df = _m5_bars([{
            "dt": datetime(2026, 5, 19, 8, 0, tzinfo=timezone.utc),
            "open": 3210.0, "high": 3215.0, "low": 3205.0, "close": 3212.0,
        }])
        bias = engine.update(df)

        assert bias.phase == SessionPhase.NO_DATA
        assert bias.direction is None
        assert bias.confidence_modifier == 0.0


# ---------------------------------------------------------------------------
# Full daily cycle
# ---------------------------------------------------------------------------

class TestFullDailyCycle:
    def test_asian_accumulation_to_london_distribution(self, monkeypatch):
        """Simulate a complete daily Po3 cycle: Asian range → London sweep → distribution."""
        monkeypatch.setenv("PRISM_SESSION_DISPLACEMENT_PIPS", "10")
        ref = date(2026, 5, 19)
        engine = SessionBiasEngine("XAUUSD")

        # Step 1: Asian bars form a 3200–3220 range
        asian = _asian_bars(ref, n=72, base_price=3200.0, range_size=20.0)
        asian_df = _m5_bars(asian)
        ar = engine.load_asian_range(asian_df, ref_date=ref)
        assert ar is not None

        # Step 2: London opens, sweeps the low (3200), then reverses up
        london_bars = [
            # Sweep bar: wick below 3200, close back above
            {"dt": datetime(2026, 5, 19, 7, 30, tzinfo=timezone.utc),
             "open": 3203.0, "high": 3205.0, "low": 3195.0, "close": 3204.0},
            # Distribution: price moves up 15+ pips from sweep low
            {"dt": datetime(2026, 5, 19, 7, 35, tzinfo=timezone.utc),
             "open": 3204.0, "high": 3215.0, "low": 3203.0, "close": 3213.0},
        ]

        full_df = _m5_bars(asian + london_bars)
        bias = engine.update(full_df)

        assert bias.sweep_side == "LOW"
        assert bias.direction == "LONG"
        assert bias.displacement_confirmed is True
        assert bias.phase == SessionPhase.DISTRIBUTION
        assert bias.confidence_modifier == 1.0
        assert bias.asian_range is not None
        assert bias.asian_range.date == ref
