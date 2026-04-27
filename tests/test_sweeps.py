"""Phase 6.C — SweepDetector unit tests."""

import pandas as pd
import pytest

from prism.signal.sweeps import LiquiditySweep, SweepDetector


def _row(o: float, h: float, l: float, c: float) -> dict:
    return {"open": o, "high": h, "low": l, "close": c}


def _flat_baseline(n: int, low: float = 1.1000, high: float = 1.1010) -> list[dict]:
    """Bars that all sit inside [low, high]."""
    return [_row(low, high, low, low + (high - low) / 2) for _ in range(n)]


def test_detect_high_sweep():
    rows = _flat_baseline(20, 1.1000, 1.1010)
    # Bar 20 wicks above 1.1010 then closes back below
    rows.append(_row(1.1009, 1.1025, 1.1008, 1.1005))
    # Follow-through down to satisfy displacement (>=5 pips below sweep close 1.1005)
    rows.append(_row(1.1005, 1.1006, 1.0998, 1.0999))
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    sweeps = det.detect(df)
    assert any(s.type == "HIGH_SWEEP" and s.sweep_bar == 20 for s in sweeps)
    s = next(s for s in sweeps if s.type == "HIGH_SWEEP")
    assert s.swept_level == 1.1010
    assert s.close_inside is True
    assert s.displacement_followed is True


def test_detect_low_sweep():
    rows = _flat_baseline(20, 1.1000, 1.1010)
    rows.append(_row(1.1002, 1.1003, 1.0985, 1.1004))
    rows.append(_row(1.1004, 1.1012, 1.1003, 1.1011))  # +7 pips above sweep close
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    sweeps = det.detect(df)
    s = next(s for s in sweeps if s.type == "LOW_SWEEP")
    assert s.swept_level == 1.1000
    assert s.sweep_bar == 20
    assert s.displacement_followed is True


def test_sweep_requires_close_inside():
    """Wick above + close also above (true breakout) is NOT a sweep."""
    rows = _flat_baseline(20, 1.1000, 1.1010)
    rows.append(_row(1.1009, 1.1025, 1.1008, 1.1020))  # closes ABOVE swept level
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    sweeps = det.detect(df)
    assert all(s.sweep_bar != 20 for s in sweeps)


def test_has_recent_sweep_long_requires_low_sweep():
    rows = _flat_baseline(20, 1.1000, 1.1010)
    rows.append(_row(1.1002, 1.1003, 1.0985, 1.1004))
    rows.append(_row(1.1004, 1.1012, 1.1003, 1.1011))
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    det.detect(df)
    assert det.has_recent_sweep("LONG", bars_back=5) is True
    assert det.has_recent_sweep("SHORT", bars_back=5) is False


def test_has_recent_sweep_short_requires_high_sweep():
    rows = _flat_baseline(20, 1.1000, 1.1010)
    rows.append(_row(1.1009, 1.1025, 1.1008, 1.1005))
    rows.append(_row(1.1005, 1.1006, 1.0998, 1.0999))
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    det.detect(df)
    assert det.has_recent_sweep("SHORT", bars_back=5) is True
    assert det.has_recent_sweep("LONG", bars_back=5) is False


def test_no_sweep_returns_false():
    rows = _flat_baseline(25, 1.1000, 1.1010)
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    det.detect(df)
    assert det.has_recent_sweep("LONG", bars_back=5) is False
    assert det.has_recent_sweep("SHORT", bars_back=5) is False
    assert det.last_sweep("LONG") is None


def test_sweep_displacement_required():
    """Sweep without follow-through fails has_recent_sweep when require_displacement=True."""
    rows = _flat_baseline(20, 1.1000, 1.1010)
    # Sweep but no displacement after (next bar drifts back into the range)
    rows.append(_row(1.1002, 1.1003, 1.0985, 1.1004))
    rows.append(_row(1.1004, 1.1006, 1.1003, 1.1005))  # only +1 pip above sweep close
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    det.detect(df)
    s = det.last_sweep("LONG")
    assert s is not None and s.displacement_followed is False
    assert det.has_recent_sweep("LONG", bars_back=5) is False
    assert det.has_recent_sweep("LONG", bars_back=5, require_displacement=False) is True


def test_last_sweep_returns_most_recent_matching():
    rows = _flat_baseline(20, 1.1000, 1.1010)
    rows.append(_row(1.1009, 1.1025, 1.1008, 1.1005))  # HIGH at 20
    rows.append(_row(1.1005, 1.1006, 1.0998, 1.0999))
    rows.append(_row(1.0999, 1.1000, 1.0997, 1.0998))
    rows.append(_row(1.1000, 1.1030, 1.0999, 1.1005))  # HIGH at 23
    rows.append(_row(1.1005, 1.1006, 1.0998, 1.0999))
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    det.detect(df)
    last = det.last_sweep("SHORT")
    assert last is not None
    assert last.sweep_bar == 23


def test_detect_idempotent_across_calls():
    rows = _flat_baseline(20, 1.1000, 1.1010)
    rows.append(_row(1.1009, 1.1025, 1.1008, 1.1005))
    rows.append(_row(1.1005, 1.1006, 1.0998, 1.0999))
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    first = det.detect(df)
    n1 = len(det.sweeps)
    second = det.detect(df)
    assert first
    assert second == []
    assert len(det.sweeps) == n1


def test_detect_empty_or_short_df():
    det = SweepDetector("EURUSD", lookback=20)
    assert det.detect(pd.DataFrame()) == []
    df = pd.DataFrame(_flat_baseline(15, 1.10, 1.11))
    assert det.detect(df) == []  # len <= lookback


def test_detect_requires_ohlc_columns():
    det = SweepDetector("EURUSD", lookback=5)
    with pytest.raises(ValueError, match="open"):
        det.detect(pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7]}))


def test_invalid_direction_returns_safe_defaults():
    rows = _flat_baseline(20, 1.1000, 1.1010)
    rows.append(_row(1.1002, 1.1003, 1.0985, 1.1004))
    rows.append(_row(1.1004, 1.1012, 1.1003, 1.1011))
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    det.detect(df)
    assert det.has_recent_sweep("DIAGONAL", bars_back=5) is False
    assert det.last_sweep("SIDEWAYS") is None


def test_bars_back_window_excludes_old_sweeps():
    rows = _flat_baseline(20, 1.1000, 1.1010)
    rows.append(_row(1.1002, 1.1003, 1.0985, 1.1004))  # bar 20
    rows.append(_row(1.1004, 1.1012, 1.1003, 1.1011))  # displacement +7
    # Add many bars after — sweep will be far in the past
    rows.extend(_flat_baseline(15, 1.1006, 1.1010))
    df = pd.DataFrame(rows)
    det = SweepDetector("EURUSD", lookback=20)
    det.detect(df)
    assert det.has_recent_sweep("LONG", bars_back=5) is False  # too old
    assert det.has_recent_sweep("LONG", bars_back=50) is True
