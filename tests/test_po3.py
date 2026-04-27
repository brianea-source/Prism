"""Phase 6.C — Po3Detector unit tests."""

import pandas as pd
import pytest

from prism.signal.po3 import DEFAULT_ACCUM_BARS, Po3Detector, Po3Phase, Po3State


def _row(o: float, h: float, l: float, c: float) -> dict:
    return {"open": o, "high": h, "low": l, "close": c}


def _accum_window(n: int = DEFAULT_ACCUM_BARS, mid: float = 1.1005) -> list[dict]:
    """Tight range — forms accumulation high/low at 1.1010 / 1.1000."""
    rows: list[dict] = []
    for i in range(n):
        rows.append(_row(mid, 1.1010, 1.1000, mid))
    return rows


def test_detect_accumulation_phase():
    rows = _accum_window()
    # Half-window only: not enough bars to leave accumulation
    df = pd.DataFrame(rows[: DEFAULT_ACCUM_BARS // 2])
    det = Po3Detector("EURUSD")
    state = det.detect_phase(df, session="LONDON")
    assert state.phase == Po3Phase.ACCUMULATION
    assert state.sweep_detected is False
    assert state.displacement_detected is False


def test_detect_manipulation_phase():
    """Sweep happens but no displacement follows → MANIPULATION."""
    rows = _accum_window()
    rows.append(_row(1.1009, 1.1025, 1.1008, 1.1005))  # HIGH sweep above 1.1010
    # Subsequent bars chop, no >=15 pip move down from 1.1005
    rows.extend([_row(1.1005, 1.1007, 1.1003, 1.1004) for _ in range(3)])
    df = pd.DataFrame(rows)
    det = Po3Detector("EURUSD")
    state = det.detect_phase(df, session="LONDON")
    assert state.sweep_detected is True
    assert state.displacement_detected is False
    assert state.phase == Po3Phase.MANIPULATION
    assert det.is_entry_phase(state) is False


def test_detect_distribution_phase():
    rows = _accum_window()
    # HIGH sweep at first post-accum bar
    rows.append(_row(1.1009, 1.1025, 1.1008, 1.1005))
    # Distribution: price drives down >=15 pips (1.1005 - 1.0985 = 20 pips)
    rows.append(_row(1.1005, 1.1006, 1.0985, 1.0988))
    df = pd.DataFrame(rows)
    det = Po3Detector("EURUSD")
    state = det.detect_phase(df, session="LONDON")
    assert state.sweep_detected is True
    assert state.displacement_detected is True
    assert state.phase == Po3Phase.DISTRIBUTION


def test_is_entry_phase_requires_both():
    state = Po3State(
        phase=Po3Phase.MANIPULATION,
        session="NY",
        session_open=1.10,
        session_high=1.11,
        session_low=1.099,
        range_size_pips=110.0,
        sweep_detected=True,
        displacement_detected=False,
    )
    det = Po3Detector("EURUSD")
    assert det.is_entry_phase(state) is False
    state.displacement_detected = True
    assert det.is_entry_phase(state) is True


def test_po3_session_label_propagates():
    rows = _accum_window()
    df = pd.DataFrame(rows[:5])
    det = Po3Detector("EURUSD")
    state = det.detect_phase(df, session="NY")
    assert state.session == "NY"


def test_po3_invalidation_when_sweep_without_displacement_fails_entry():
    rows = _accum_window()
    rows.append(_row(1.1002, 1.1003, 1.0985, 1.1004))  # LOW sweep
    rows.append(_row(1.1004, 1.1006, 1.1003, 1.1005))  # +1 pip up — no displacement
    df = pd.DataFrame(rows)
    det = Po3Detector("EURUSD")
    state = det.detect_phase(df, session="LONDON")
    assert state.phase == Po3Phase.MANIPULATION
    assert det.is_entry_phase(state) is False


def test_detect_empty_returns_unknown():
    det = Po3Detector("EURUSD")
    state = det.detect_phase(pd.DataFrame(), session="LONDON")
    assert state.phase == Po3Phase.UNKNOWN
    assert state.session == "LONDON"


def test_detect_requires_ohlc_columns():
    det = Po3Detector("EURUSD")
    with pytest.raises(ValueError, match="open"):
        det.detect_phase(pd.DataFrame({"x": [1, 2, 3]}), session="LONDON")
