"""Phase 6.D integration tests — SignalGenerator wires OB / Sweep / Po3.

These tests reuse the same mock plumbing as ``test_phase2.py`` to drive
``SignalGenerator.generate()`` end-to-end and assert the new smart-money
behaviour. Detectors are mocked at the *instance* level so we don't need to
hand-craft OB/sweep/Po3 dataframes — the goal here is the *integration* path,
not detector-internals (those are covered by their own test files).

Coverage:
* master switch off → ``packet.smart_money is None`` (default behaviour)
* master switch on  → packet populated with ``ob``/``sweep``/``po3`` dicts
* sweep gate blocks signal when no qualifying sweep
* Po3 gate blocks signal when not in entry phase
* both gates disabled → signal passes regardless of smart-money state
* Slack notifier renders the smart-money block when present
"""

from __future__ import annotations

import datetime as _dt
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from prism.delivery.slack_notifier import SlackNotifier
from prism.execution.mt5_bridge import SignalPacket
from prism.news.intelligence import NewsSignal
from prism.signal.htf_bias import Bias
from prism.signal.order_blocks import OrderBlock, OrderBlockState
from prism.signal.po3 import Po3Phase, Po3State
from prism.signal.sweeps import LiquiditySweep


# ---------------------------------------------------------------------------
# Shared fixtures — replica of test_phase2.py's H4 bullish-FVG builder.
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
    return gen


def _stub_smart_money_detectors(
    gen,
    *,
    has_recent_sweep: bool = True,
    is_entry_phase: bool = True,
    nearest_ob: bool = True,
    sweep: bool = True,
):
    """Replace detector methods with deterministic mocks for integration tests."""
    ob = (
        OrderBlock(
            instrument="EURUSD", timeframe="H4", direction="BULLISH",
            high=1.1015, low=1.1005, midpoint=1.1010,
            formed_at="2024-01-01T08:00:00", formed_bar=12,
            displacement_size=20.0, state=OrderBlockState.OB_FRESH,
        )
        if nearest_ob
        else None
    )
    gen.ob_detector.detect = MagicMock(return_value=[])
    gen.ob_detector.update_states = MagicMock(return_value=None)
    gen.ob_detector.get_nearest_ob = MagicMock(return_value=ob)
    gen.ob_detector.distance_to_ob = MagicMock(return_value=15.0 if ob else None)

    last_sweep = (
        LiquiditySweep(
            instrument="EURUSD", type="LOW_SWEEP", swept_level=1.0998,
            sweep_bar=27, close_inside=True, timestamp="bar-27",
            displacement_followed=has_recent_sweep,
        )
        if sweep
        else None
    )
    gen.sweep_detector.detect = MagicMock(return_value=[])
    gen.sweep_detector.last_sweep = MagicMock(return_value=last_sweep)
    gen.sweep_detector.has_recent_sweep = MagicMock(return_value=has_recent_sweep)
    gen.sweep_detector._latest_scanned_bar = 30 if last_sweep else None

    po3_state = Po3State(
        phase=Po3Phase.DISTRIBUTION if is_entry_phase else Po3Phase.MANIPULATION,
        session="London Kill Zone",
        session_open=1.1000, session_high=1.1060, session_low=1.0990,
        range_size_pips=70.0,
        sweep_detected=is_entry_phase or True,
        displacement_detected=is_entry_phase,
    )
    gen.po3_detector.detect_phase = MagicMock(return_value=po3_state)
    gen.po3_detector.is_entry_phase = MagicMock(return_value=is_entry_phase)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_smart_money_disabled_by_default(tmp_path, monkeypatch):
    """Master switch OFF → packet.smart_money is None and detectors aren't called."""
    gen = _make_generator(tmp_path, monkeypatch)
    _stub_smart_money_detectors(gen)

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PRISM_SMART_MONEY_ENABLED", None)
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is not None
    assert packet.smart_money is None
    gen.ob_detector.detect.assert_not_called()
    gen.sweep_detector.detect.assert_not_called()
    gen.po3_detector.detect_phase.assert_not_called()


def test_smart_money_enabled_populates_packet(tmp_path, monkeypatch):
    """Master ON, all detectors permissive → packet carries OB/sweep/Po3 dicts."""
    gen = _make_generator(tmp_path, monkeypatch)
    _stub_smart_money_detectors(gen)

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "1",
        "PRISM_PO3_REQUIRED": "1",
    }):
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is not None
    assert packet.smart_money is not None
    sm = packet.smart_money

    assert "blocked" not in sm  # gate keys are stripped from the packet payload
    assert "block_reason" not in sm

    assert sm["ob"] is not None
    assert sm["ob"]["state"] == "OB_FRESH"
    assert sm["ob"]["effective_direction"] == "BULLISH"
    assert sm["ob"]["distance_pips"] == 15.0
    assert sm["ob"]["in_range"] is True

    assert sm["sweep"] is not None
    assert sm["sweep"]["type"] == "LOW_SWEEP"
    assert sm["sweep"]["bars_ago"] == 3  # latest_scanned 30 - sweep_bar 27
    assert sm["sweep"]["qualifies"] is True

    assert sm["po3"] is not None
    assert sm["po3"]["phase"] == "DISTRIBUTION"
    assert sm["po3"]["is_entry_phase"] is True


def test_smart_money_sweep_required_blocks_when_no_recent_sweep(tmp_path, monkeypatch):
    """PRISM_SWEEP_REQUIRED=1 + no qualifying sweep → generate() returns None."""
    gen = _make_generator(tmp_path, monkeypatch)
    _stub_smart_money_detectors(gen, has_recent_sweep=False, sweep=False)

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "1",
        "PRISM_PO3_REQUIRED": "0",
    }):
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is None


def test_smart_money_po3_required_blocks_when_not_entry_phase(tmp_path, monkeypatch):
    """Sweep present but Po3 not in DISTRIBUTION → blocked when PO3_REQUIRED=1."""
    gen = _make_generator(tmp_path, monkeypatch)
    _stub_smart_money_detectors(gen, has_recent_sweep=True, is_entry_phase=False)

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "1",
        "PRISM_PO3_REQUIRED": "1",
    }):
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is None


def test_smart_money_gates_disabled_passes_with_observability(tmp_path, monkeypatch):
    """Master ON but both gates off → packet still built; smart_money populated."""
    gen = _make_generator(tmp_path, monkeypatch)
    _stub_smart_money_detectors(
        gen, has_recent_sweep=False, is_entry_phase=False, nearest_ob=False, sweep=False
    )

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
    assert packet.smart_money is not None
    assert packet.smart_money["ob"] is None
    assert packet.smart_money["sweep"] is None
    assert packet.smart_money["po3"] is not None  # detect_phase still ran
    assert packet.smart_money["po3"]["is_entry_phase"] is False


def test_smart_money_ob_distance_outside_range_marks_not_in_range(tmp_path, monkeypatch):
    """An OB beyond PRISM_OB_MAX_DISTANCE_PIPS is reported but flagged in_range=False."""
    gen = _make_generator(tmp_path, monkeypatch)
    _stub_smart_money_detectors(gen)
    gen.ob_detector.distance_to_ob = MagicMock(return_value=80.0)

    with patch.dict(os.environ, {
        "PRISM_SMART_MONEY_ENABLED": "1",
        "PRISM_SWEEP_REQUIRED": "0",
        "PRISM_PO3_REQUIRED": "0",
        "PRISM_OB_MAX_DISTANCE_PIPS": "30",
    }):
        packet = gen.generate(
            h4_df=_h4_bullish_fvg(),
            h1_df=_h4_bullish_fvg(),
            entry_df=_entry_df_into_zone(),
        )

    assert packet is not None
    ob = packet.smart_money["ob"]
    assert ob["distance_pips"] == 80.0
    assert ob["in_range"] is False


def test_slack_notifier_renders_smart_money_block():
    """SlackNotifier formats OB/sweep/Po3 lines when smart_money is populated."""
    notifier = SlackNotifier(token="", channel="#test")
    packet = SignalPacket(
        instrument="EURUSD", direction="LONG",
        entry=1.1020, sl=1.0990, tp1=1.1080, tp2=1.1110,
        rr_ratio=2.5, confidence=0.78, confidence_level="HIGH",
        magnitude_pips=35.0, regime="NEUTRAL", news_bias="BULLISH",
        fvg_zone={"timeframe": "H4", "bottom": 1.1000, "top": 1.1030,
                  "partially_mitigated": False},
        signal_time="2024-01-01T12:00:00",
        smart_money={
            "ob": {
                "state": "OB_FRESH", "direction": "BULLISH",
                "effective_direction": "BULLISH",
                "high": 1.1015, "low": 1.1005, "midpoint": 1.1010,
                "timeframe": "H4", "distance_pips": 12.5,
                "is_rejection_block": False, "in_range": True,
            },
            "sweep": {
                "type": "LOW_SWEEP", "swept_level": 1.0998, "sweep_bar": 27,
                "bars_ago": 3, "displacement_followed": True,
                "timestamp": "bar-27", "qualifies": True,
            },
            "po3": {
                "phase": "DISTRIBUTION", "session": "London Kill Zone",
                "range_size_pips": 70.0, "sweep_detected": True,
                "displacement_detected": True, "is_entry_phase": True,
            },
        },
    )

    blocks = notifier._format_signal_blocks(packet)
    body = blocks[0]["text"]["text"]

    assert "Smart Money" in body
    assert "OB BULLISH H4" in body
    assert "12.5 pips away" in body
    assert "LOW_SWEEP" in body
    assert "3 bars ago" in body
    assert "DISTRIBUTION" in body
    assert "London Kill Zone" in body


def test_slack_notifier_omits_smart_money_when_absent():
    """No smart_money field → no Smart Money section rendered."""
    notifier = SlackNotifier(token="", channel="#test")
    packet = SignalPacket(
        instrument="EURUSD", direction="LONG",
        entry=1.1020, sl=1.0990, tp1=1.1080, tp2=1.1110,
        rr_ratio=2.5, confidence=0.78, confidence_level="HIGH",
        magnitude_pips=35.0, regime="NEUTRAL", news_bias="BULLISH",
        fvg_zone={"timeframe": "H4", "bottom": 1.1000, "top": 1.1030,
                  "partially_mitigated": False},
        signal_time="2024-01-01T12:00:00",
        smart_money=None,
    )

    blocks = notifier._format_signal_blocks(packet)
    body = blocks[0]["text"]["text"]
    assert "Smart Money" not in body


def test_slack_notifier_smart_money_none_subdicts_show_fallback_lines():
    """When smart_money is set but every sub-dict is None, show 'none' fallbacks."""
    notifier = SlackNotifier(token="", channel="#test")
    packet = SignalPacket(
        instrument="EURUSD", direction="LONG",
        entry=1.1020, sl=1.0990, tp1=1.1080, tp2=1.1110,
        rr_ratio=2.5, confidence=0.78, confidence_level="HIGH",
        magnitude_pips=35.0, regime="NEUTRAL", news_bias="BULLISH",
        fvg_zone={"timeframe": "H4", "bottom": 1.1000, "top": 1.1030,
                  "partially_mitigated": False},
        signal_time="2024-01-01T12:00:00",
        smart_money={"ob": None, "sweep": None, "po3": None},
    )

    blocks = notifier._format_signal_blocks(packet)
    body = blocks[0]["text"]["text"]
    assert "Smart Money" in body
    assert "OB: none in range" in body
    assert "Sweep: none recent" in body
    assert "Po3: unknown" in body
