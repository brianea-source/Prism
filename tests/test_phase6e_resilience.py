"""Phase 6.E resilience tests — smart-money detectors must fail closed and loud.

A detector that crashes silently in production is the exact failure mode the
observability rollout is supposed to *catch*, not paper over. These tests pin
down the contract:

1. **Fail closed.** When ``PRISM_SWEEP_REQUIRED=1`` or ``PRISM_PO3_REQUIRED=1``
   and the underlying detector raises, ``generate()`` returns ``None``. We do
   NOT let signals through with missing confluence.

2. **Fail loud.** Every detector exception increments
   ``gen.detector_failure_counts[<name>]`` and logs at ``ERROR`` with the full
   traceback (``exc_info=True``).

3. **OB is observability-only.** An OB exception never blocks a signal — the
   ``ob`` sub-dict is set to ``None`` and the packet is still produced.

The fixtures here re-use the same builder pattern as
``test_phase6d_integration.py`` so the tests stay readable.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from prism.news.intelligence import NewsSignal
from prism.signal.htf_bias import Bias


# ---------------------------------------------------------------------------
# Builders (kept local so 6.D and 6.E test files can evolve independently).
# ---------------------------------------------------------------------------

def _h4_bullish_fvg(n: int = 40, base: float = 1.1000) -> pd.DataFrame:
    rows: list[dict] = []
    ts = datetime(2024, 1, 1)

    for i in range(n - 10):
        rows.append({
            "datetime": ts + _dt.timedelta(hours=4 * i),
            "open": base, "high": base + 0.0005, "low": base - 0.0005,
            "close": base, "volume": 100, "atr_14": 0.001,
        })

    rows[-1]["high"] = 1.1000
    rows[-1]["low"] = 1.0990
    rows[-1]["close"] = 1.1000
    rows.append({
        "datetime": ts + _dt.timedelta(hours=4 * len(rows)),
        "open": 1.1010, "high": 1.1040, "low": 1.1005, "close": 1.1035,
        "volume": 200, "atr_14": 0.001,
    })
    rows.append({
        "datetime": ts + _dt.timedelta(hours=4 * len(rows)),
        "open": 1.1035, "high": 1.1060, "low": 1.1030, "close": 1.1050,
        "volume": 200, "atr_14": 0.001,
    })

    for _ in range(8):
        rows.append({
            "datetime": ts + _dt.timedelta(hours=4 * len(rows)),
            "open": 1.1020, "high": 1.1045, "low": 1.1016, "close": 1.1020,
            "volume": 150, "atr_14": 0.001,
        })

    df = pd.DataFrame(rows)
    df["rsi_14"] = 55.0
    return df


def _entry_df_into_zone() -> pd.DataFrame:
    rows = [
        {"open": 1.1055, "high": 1.1065, "low": 1.1045, "close": 1.1060},
        {"open": 1.1055, "high": 1.1065, "low": 1.1045, "close": 1.1060},
        {"open": 1.1055, "high": 1.1065, "low": 1.1045, "close": 1.1060},
        {"open": 1.1035, "high": 1.1040, "low": 1.1025, "close": 1.1030},
    ]
    df = pd.DataFrame(rows)
    df["atr_14"] = 0.001
    return df


def _make_generator(tmp_path, monkeypatch):
    import prism.signal.fvg as fvg_mod
    from prism.signal.generator import SignalGenerator

    monkeypatch.setattr(fvg_mod, "FVG_STORE_DIR", tmp_path)
    gen = SignalGenerator("EURUSD")

    gen.news.get_signal = MagicMock(return_value=NewsSignal(
        instrument="EURUSD", timestamp="2024-01-01T12:00:00",
        news_bias="BULLISH", event_flag=False, event_name="",
        risk_regime="NEUTRAL", sentiment_score=0.3,
        geopolitical_active=False, sources=[],
    ))
    gen.news.should_block_trade = MagicMock(return_value=(False, ""))

    gen.htf_engine.refresh = MagicMock(return_value=MagicMock(
        bias_1h=Bias.BULLISH, bias_4h=Bias.BULLISH,
        aligned=True, allowed_direction="LONG",
        swing_points_1h=[
            {"type": "HH", "price": 1.10, "bar_idx": 5},
            {"type": "HL", "price": 1.09, "bar_idx": 10},
            {"type": "HH", "price": 1.11, "bar_idx": 15},
        ],
        swing_points_4h=[
            {"type": "HH", "price": 1.10, "bar_idx": 3},
            {"type": "HL", "price": 1.09, "bar_idx": 6},
            {"type": "HH", "price": 1.12, "bar_idx": 9},
        ],
    ))
    gen.htf_engine.gate_signal = MagicMock(return_value=(True, "HTF aligned"))

    mock_predictor = MagicMock()
    mock_predictor.predict_latest.return_value = {
        "direction": 1, "direction_str": "LONG", "confidence": 0.78,
        "confidence_level": "HIGH", "magnitude_pips": 35.0,
    }
    gen._predictor = mock_predictor

    gen.icc.detect_signals = MagicMock(return_value=[{
        "phase": "CONTINUATION", "direction": "LONG",
        "entry": 1.1020, "sl": 1.0990, "correction_low": 1.0995,
        "leg_size": 0.0060, "correction_pct": 0.4,
        "indication_range": 0.0050, "indication_range_pips": 50.0,
        "aoi_nearby": False,
    }])

    # Default: sweep + Po3 detectors are happy. Tests override individually.
    gen.sweep_detector.detect = MagicMock(return_value=[])
    gen.sweep_detector.last_sweep = MagicMock(return_value=None)
    gen.sweep_detector.has_recent_sweep = MagicMock(return_value=True)
    gen.sweep_detector._latest_scanned_bar = 30

    from prism.signal.po3 import Po3Phase, Po3State
    gen.po3_detector.detect_phase = MagicMock(return_value=Po3State(
        phase=Po3Phase.DISTRIBUTION, session="London Kill Zone",
        session_open=1.1000, session_high=1.1060, session_low=1.0990,
        range_size_pips=70.0, sweep_detected=True, displacement_detected=True,
    ))
    gen.po3_detector.is_entry_phase = MagicMock(return_value=True)

    gen.ob_detector.detect = MagicMock(return_value=[])
    gen.ob_detector.update_states = MagicMock(return_value=None)
    gen.ob_detector.get_nearest_ob = MagicMock(return_value=None)
    gen.ob_detector.distance_to_ob = MagicMock(return_value=None)

    return gen


# ---------------------------------------------------------------------------
# Failure counter init
# ---------------------------------------------------------------------------

def test_detector_failure_counts_initialised_to_zero():
    """Fresh SignalGenerator exposes a zeroed failure counter for ops to scrape."""
    from prism.signal.generator import SignalGenerator
    gen = SignalGenerator("EURUSD", persist_fvg=False)
    assert gen.detector_failure_counts == {"ob": 0, "sweep": 0, "po3": 0}


# ---------------------------------------------------------------------------
# OB exception — observability-only, packet still built
# ---------------------------------------------------------------------------

def test_ob_detector_exception_does_not_block_signal(tmp_path, monkeypatch, caplog):
    """OB has no gate — an exception is logged + counted but the packet still ships."""
    gen = _make_generator(tmp_path, monkeypatch)
    gen.ob_detector.detect = MagicMock(side_effect=RuntimeError("boom-ob"))

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "0",
        "PRISM_PO3_REQUIRED": "0",
    }), caplog.at_level(logging.ERROR, logger="prism.signal.generator"):
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is not None
    assert packet.smart_money is not None
    assert packet.smart_money["ob"] is None
    assert gen.detector_failure_counts["ob"] == 1
    assert gen.detector_failure_counts["sweep"] == 0
    assert gen.detector_failure_counts["po3"] == 0

    error_records = [
        r for r in caplog.records
        if r.levelno == logging.ERROR and "OrderBlockDetector failed" in r.getMessage()
    ]
    assert error_records, "expected an ERROR log for the OB exception"
    assert error_records[0].exc_info is not None, "logger.error must include the traceback"


# ---------------------------------------------------------------------------
# Sweep exception — fail closed when required, fail open when not
# ---------------------------------------------------------------------------

def test_sweep_detector_exception_blocks_when_required(tmp_path, monkeypatch, caplog):
    """A SweepDetector crash with PRISM_SWEEP_REQUIRED=1 must block the signal."""
    gen = _make_generator(tmp_path, monkeypatch)
    gen.sweep_detector.detect = MagicMock(side_effect=RuntimeError("boom-sweep"))

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "1",
        "PRISM_PO3_REQUIRED": "0",
    }), caplog.at_level(logging.ERROR, logger="prism.signal.generator"):
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is None, "fail-closed: broken sweep detector must not let trades through"
    assert gen.detector_failure_counts["sweep"] == 1
    assert any(
        "SweepDetector failed" in r.getMessage() and r.levelno == logging.ERROR
        for r in caplog.records
    )


def test_sweep_detector_exception_passes_when_not_required(tmp_path, monkeypatch):
    """With PRISM_SWEEP_REQUIRED=0 the broken detector is observability-only."""
    gen = _make_generator(tmp_path, monkeypatch)
    gen.sweep_detector.detect = MagicMock(side_effect=RuntimeError("boom-sweep"))

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "0",
        "PRISM_PO3_REQUIRED": "0",
    }):
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is not None
    assert packet.smart_money["sweep"] is None
    assert gen.detector_failure_counts["sweep"] == 1


# ---------------------------------------------------------------------------
# Po3 exception — fail closed when required, fail open when not
# ---------------------------------------------------------------------------

def test_po3_detector_exception_blocks_when_required(tmp_path, monkeypatch, caplog):
    """Po3 crash with PRISM_PO3_REQUIRED=1 must block the signal."""
    gen = _make_generator(tmp_path, monkeypatch)
    gen.po3_detector.detect_phase = MagicMock(side_effect=RuntimeError("boom-po3"))

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "0",
        "PRISM_PO3_REQUIRED": "1",
    }), caplog.at_level(logging.ERROR, logger="prism.signal.generator"):
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is None
    assert gen.detector_failure_counts["po3"] == 1
    assert any(
        "Po3Detector failed" in r.getMessage() and r.levelno == logging.ERROR
        for r in caplog.records
    )


def test_po3_detector_exception_passes_when_not_required(tmp_path, monkeypatch):
    """Observability-only mode tolerates a Po3 crash without blocking."""
    gen = _make_generator(tmp_path, monkeypatch)
    gen.po3_detector.detect_phase = MagicMock(side_effect=RuntimeError("boom-po3"))

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "0",
        "PRISM_PO3_REQUIRED": "0",
    }):
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is not None
    assert packet.smart_money["po3"] is None
    assert gen.detector_failure_counts["po3"] == 1


# ---------------------------------------------------------------------------
# Failure counter is per-instance and accumulates across calls
# ---------------------------------------------------------------------------

def test_failure_counter_accumulates_across_generate_calls(tmp_path, monkeypatch):
    """Each crash bumps the counter; counters are not reset between bars."""
    gen = _make_generator(tmp_path, monkeypatch)
    gen.ob_detector.detect = MagicMock(side_effect=RuntimeError("boom-ob"))

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "0",
        "PRISM_PO3_REQUIRED": "0",
    }):
        for _ in range(3):
            gen.generate(
                h4_df=_h4_bullish_fvg(),
                h1_df=_h4_bullish_fvg(),
                entry_df=_entry_df_into_zone(),
            )

    assert gen.detector_failure_counts["ob"] == 3
    assert gen.detector_failure_counts["sweep"] == 0
    assert gen.detector_failure_counts["po3"] == 0
