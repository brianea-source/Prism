"""
PRISM Phase 2 — Test Suite
Tests: FVG detector, news intelligence, MT5 mock bridge, signal generator news blocking.
"""
import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.signal.fvg import FVGDetector, FVGZone
from prism.news.intelligence import NewsIntelligence, NewsSignal
from prism.execution.mt5_bridge import MockMT5Bridge, SignalPacket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_df(n: int = 10, base: float = 1.1000) -> pd.DataFrame:
    """Flat candles — no gaps."""
    rows = []
    ts = datetime(2024, 1, 1)
    for i in range(n):
        rows.append({
            "datetime": ts + __import__("datetime").timedelta(hours=i),
            "open":  base,
            "high":  base + 0.0005,
            "low":   base - 0.0005,
            "close": base,
            "volume": 100,
        })
    return pd.DataFrame(rows)


def _make_signal_packet(
    instrument="XAUUSD",
    direction="LONG",
    entry=1900.0,
    sl=1890.0,
    confidence=0.75,
) -> SignalPacket:
    return SignalPacket(
        instrument=instrument,
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=entry + 15,
        tp2=entry + 25,
        rr_ratio=2.5,
        confidence=confidence,
        confidence_level="HIGH",
        magnitude_pips=25.0,
        regime="NEUTRAL",
        news_bias="BULLISH",
        fvg_zone=None,
        signal_time=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# FVG Tests
# ---------------------------------------------------------------------------

def test_fvg_bullish_detected():
    """
    3-bar setup: candle[0].high < candle[2].low  →  BULLISH FVG.
    """
    rows = [
        {"datetime": datetime(2024, 1, 1, 0), "open": 1.10, "high": 1.10, "low": 1.09, "close": 1.10, "volume": 100},
        {"datetime": datetime(2024, 1, 1, 1), "open": 1.11, "high": 1.12, "low": 1.10, "close": 1.12, "volume": 200},  # impulse
        {"datetime": datetime(2024, 1, 1, 2), "open": 1.13, "high": 1.14, "low": 1.115, "close": 1.13, "volume": 150},
        # candle[0].high=1.10 < candle[2].low=1.115  →  gap exists
    ]
    df = pd.DataFrame(rows)
    detector = FVGDetector("EURUSD", "H4")
    zones = detector.detect(df)

    bullish = [z for z in zones if z.direction == "BULLISH"]
    assert len(bullish) >= 1, f"Expected at least 1 BULLISH FVG, got: {zones}"
    z = bullish[0]
    assert z.bottom == pytest.approx(1.10, abs=1e-5)
    assert z.top == pytest.approx(1.115, abs=1e-5)
    assert z.midline == pytest.approx((1.10 + 1.115) / 2, abs=1e-5)


def test_fvg_bearish_detected():
    """
    3-bar setup: candle[0].low > candle[2].high  →  BEARISH FVG.
    """
    rows = [
        {"datetime": datetime(2024, 1, 1, 0), "open": 1.14, "high": 1.145, "low": 1.135, "close": 1.14, "volume": 100},
        {"datetime": datetime(2024, 1, 1, 1), "open": 1.13, "high": 1.135, "low": 1.12, "close": 1.12, "volume": 200},  # impulse
        {"datetime": datetime(2024, 1, 1, 2), "open": 1.11, "high": 1.115, "low": 1.10, "close": 1.11, "volume": 150},
        # candle[0].low=1.135 > candle[2].high=1.115  →  gap exists
    ]
    df = pd.DataFrame(rows)
    detector = FVGDetector("EURUSD", "H4")
    zones = detector.detect(df)

    bearish = [z for z in zones if z.direction == "BEARISH"]
    assert len(bearish) >= 1, f"Expected at least 1 BEARISH FVG, got: {zones}"
    z = bearish[0]
    assert z.top == pytest.approx(1.135, abs=1e-5)
    assert z.bottom == pytest.approx(1.115, abs=1e-5)


def test_fvg_mitigation():
    """
    After a BULLISH FVG is detected, price closes below zone.bottom → mitigated=True.
    """
    # Build df with bullish FVG at bars 0-2, then mitigating bar at bar 3
    rows = [
        {"datetime": datetime(2024, 1, 1, 0), "open": 1.10, "high": 1.100, "low": 1.090, "close": 1.10, "volume": 100},
        {"datetime": datetime(2024, 1, 1, 1), "open": 1.11, "high": 1.120, "low": 1.100, "close": 1.12, "volume": 200},
        {"datetime": datetime(2024, 1, 1, 2), "open": 1.13, "high": 1.140, "low": 1.115, "close": 1.13, "volume": 150},
        # bar 3: close falls below zone.bottom (1.100)
        {"datetime": datetime(2024, 1, 1, 3), "open": 1.11, "high": 1.110, "low": 1.085, "close": 1.085, "volume": 300},
    ]
    df = pd.DataFrame(rows)
    detector = FVGDetector("EURUSD", "H4")
    zones = detector.detect(df)

    bullish = [z for z in zones if z.direction == "BULLISH"]
    assert len(bullish) >= 1, "Expected bullish FVG zone"
    assert bullish[0].mitigated is True, (
        f"Zone should be mitigated after close={rows[3]['close']} < bottom={bullish[0].bottom}"
    )


# ---------------------------------------------------------------------------
# News Intelligence Tests
# ---------------------------------------------------------------------------

def test_news_keyword_sentiment_gold_positive():
    """
    'gold rally safe haven' should score positive for XAUUSD.
    """
    ni = NewsIntelligence()
    score = ni._keyword_sentiment("gold rally safe haven", "XAUUSD")
    assert score > 0, f"Expected positive sentiment for XAUUSD on 'safe haven' text, got {score}"


def test_news_keyword_sentiment_neutral():
    """
    Neutral text with no keywords should return 0.0.
    """
    ni = NewsIntelligence()
    score = ni._keyword_sentiment("market update april 2024", "EURUSD")
    assert score == 0.0


def test_news_bias_risk_off_gold_bullish():
    """RISK_OFF regime → BULLISH bias for XAUUSD."""
    ni = NewsIntelligence()
    bias = ni._derive_bias(0.0, "RISK_OFF", "XAUUSD")
    assert bias == "BULLISH"


def test_news_bias_risk_on_gold_bearish():
    """RISK_ON regime → BEARISH bias for XAUUSD."""
    ni = NewsIntelligence()
    bias = ni._derive_bias(0.0, "RISK_ON", "XAUUSD")
    assert bias == "BEARISH"


def test_should_block_trade_on_event_flag():
    """event_flag=True → should_block_trade returns (True, reason)."""
    ni = NewsIntelligence()
    signal = NewsSignal(
        instrument="XAUUSD",
        timestamp="2024-01-01T12:00:00",
        news_bias="NEUTRAL",
        event_flag=True,
        event_name="NFP",
        risk_regime="NEUTRAL",
        sentiment_score=0.0,
        geopolitical_active=False,
        sources=[],
    )
    blocked, reason = ni.should_block_trade(signal)
    assert blocked is True
    assert "NFP" in reason


def test_should_not_block_trade_no_event():
    """event_flag=False → should_block_trade returns (False, '')."""
    ni = NewsIntelligence()
    signal = NewsSignal(
        instrument="XAUUSD",
        timestamp="2024-01-01T12:00:00",
        news_bias="NEUTRAL",
        event_flag=False,
        event_name="",
        risk_regime="NEUTRAL",
        sentiment_score=0.0,
        geopolitical_active=False,
        sources=[],
    )
    blocked, reason = ni.should_block_trade(signal)
    assert blocked is False


# ---------------------------------------------------------------------------
# MockMT5Bridge Tests
# ---------------------------------------------------------------------------

def test_mock_mt5_executes():
    """MockMT5Bridge.execute_signal() → success=True, ticket=99999."""
    bridge = MockMT5Bridge(mode="AUTO")
    bridge.connect()
    signal = _make_signal_packet()
    result = bridge.execute_signal(signal)
    assert result.success is True
    assert result.ticket == 99999
    assert result.error is None
    assert result.actual_entry == signal.entry
    assert result.actual_sl == signal.sl
    assert result.actual_tp == signal.tp2


def test_mock_mt5_connect_sets_connected():
    bridge = MockMT5Bridge()
    assert bridge._connected is False
    bridge.connect()
    assert bridge._connected is True


def test_mock_mt5_balance():
    bridge = MockMT5Bridge()
    bridge.connect()
    assert bridge.get_account_balance() == 1000.0


def test_mock_mt5_no_positions():
    bridge = MockMT5Bridge()
    bridge.connect()
    assert bridge.count_open_positions() == 0


# ---------------------------------------------------------------------------
# SignalGenerator — News Blocking
# ---------------------------------------------------------------------------

def test_signal_generator_blocks_on_news_event():
    """
    Patch news.should_block_trade to return (True, "NFP").
    Generator must return None.
    """
    from prism.signal.generator import SignalGenerator

    gen = SignalGenerator("XAUUSD")

    # Patch news intelligence so it blocks
    mock_news_signal = NewsSignal(
        instrument="XAUUSD",
        timestamp="2024-01-01T12:00:00",
        news_bias="NEUTRAL",
        event_flag=True,
        event_name="NFP",
        risk_regime="NEUTRAL",
        sentiment_score=0.0,
        geopolitical_active=False,
        sources=[],
    )
    gen.news.get_signal = MagicMock(return_value=mock_news_signal)
    gen.news.should_block_trade = MagicMock(return_value=(True, "High-impact event imminent: NFP"))

    # Build minimal dataframes (content doesn't matter — blocked before ML)
    df = _make_flat_df(30)
    result = gen.generate(h4_df=df, h1_df=df, entry_df=df)
    assert result is None, "Generator should return None when news blocks the trade"


def test_signal_generator_passes_when_news_clear():
    """
    When news is clear but ML model returns no signal (direction=0),
    generator should still return None (no trade, but no error).
    """
    from prism.signal.generator import SignalGenerator

    gen = SignalGenerator("XAUUSD")

    mock_news_signal = NewsSignal(
        instrument="XAUUSD",
        timestamp="2024-01-01T12:00:00",
        news_bias="NEUTRAL",
        event_flag=False,
        event_name="",
        risk_regime="NEUTRAL",
        sentiment_score=0.0,
        geopolitical_active=False,
        sources=[],
    )
    gen.news.get_signal = MagicMock(return_value=mock_news_signal)
    gen.news.should_block_trade = MagicMock(return_value=(False, ""))

    # Mock predictor to return no-trade
    mock_predictor = MagicMock()
    mock_predictor.predict_latest.return_value = {
        "direction": 0,
        "direction_str": "NONE",
        "confidence": 0.45,
        "confidence_level": "LOW",
        "magnitude_pips": 0.0,
    }
    gen._predictor = mock_predictor

    df = _make_flat_df(30)
    # Add a dummy feature column so feature_cols is non-empty
    df["rsi_14"] = 50.0

    result = gen.generate(h4_df=df, h1_df=df, entry_df=df)
    assert result is None


# ---------------------------------------------------------------------------
# Lot Size Calculation (MT5Bridge base class, exercised via Mock)
# ---------------------------------------------------------------------------

def test_lot_size_calculation_gold():
    """1% of $1000 on a 50-pip gold SL → reasonable lot size."""
    bridge = MockMT5Bridge()
    # XAUUSD: pip=0.01, 50 pips SL = 0.50 price distance
    lot = bridge.calculate_lot_size("XAUUSD", sl_price=1850.0, entry_price=1850.5, account_balance=1000.0)
    # risk_amount = $10, pip_value_per_lot=$1, sl_pips=50 → lot = 10/(50*1)=0.20
    assert lot == pytest.approx(0.20, abs=0.05)


def test_lot_size_rejects_tight_sl():
    """SL < 1 pip should return 0.0."""
    bridge = MockMT5Bridge()
    lot = bridge.calculate_lot_size("EURUSD", sl_price=1.10000, entry_price=1.10000, account_balance=1000.0)
    assert lot == 0.0


# ---------------------------------------------------------------------------
# FVG active zone filtering
# ---------------------------------------------------------------------------

def test_fvg_get_active_zones_respects_age():
    """Zones older than max_age_bars should be excluded."""
    detector = FVGDetector("EURUSD", "H4")
    detector.zones = [
        FVGZone("EURUSD", "H4", "BULLISH", 1.11, 1.10, 1.105,
                "2024-01-01", 0, age_bars=60, strength=0.5),
        FVGZone("EURUSD", "H4", "BULLISH", 1.12, 1.11, 1.115,
                "2024-01-02", 1, age_bars=10, strength=0.5),
    ]
    active = detector.get_active_zones(max_age_bars=50)
    assert len(active) == 1
    assert active[0].bottom == pytest.approx(1.11)


def test_fvg_check_entry_trigger_long():
    """Price inside BULLISH zone → check_entry_trigger returns zone for LONG."""
    detector = FVGDetector("EURUSD", "H4")
    detector.zones = [
        FVGZone("EURUSD", "H4", "BULLISH", 1.1050, 1.1000, 1.1025,
                "2024-01-01", 0, age_bars=5, strength=0.5),
    ]
    zone = detector.check_entry_trigger(1.1030, "LONG")
    assert zone is not None

    # Price above top → no trigger
    zone2 = detector.check_entry_trigger(1.1060, "LONG")
    assert zone2 is None


# ---------------------------------------------------------------------------
# New: Tiingo sentiment normalization (structured vs scalar vs missing)
# ---------------------------------------------------------------------------

def test_tiingo_sentiment_extractor_handles_compound_dict():
    """Tiingo may return sentiment as {'compound': -0.3, ...} — must extract float."""
    from prism.news.intelligence import _extract_tiingo_sentiment
    assert _extract_tiingo_sentiment({"sentiment": {"compound": -0.3}}) == pytest.approx(-0.3)


def test_tiingo_sentiment_extractor_handles_scalar():
    from prism.news.intelligence import _extract_tiingo_sentiment
    assert _extract_tiingo_sentiment({"sentiment": 0.4}) == pytest.approx(0.4)


def test_tiingo_sentiment_extractor_missing_returns_none():
    from prism.news.intelligence import _extract_tiingo_sentiment
    assert _extract_tiingo_sentiment({"title": "no sentiment here"}) is None


def test_tiingo_sentiment_extractor_pos_neg_fallback():
    """dict without 'compound' should fall back to pos - neg."""
    from prism.news.intelligence import _extract_tiingo_sentiment
    score = _extract_tiingo_sentiment({"sentiment": {"pos": 0.6, "neg": 0.1, "neu": 0.3}})
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# New: Economic calendar blackout window (T-30m through T+15m)
# ---------------------------------------------------------------------------

def test_economic_calendar_blocks_inside_window():
    """Event 10 minutes in the future must trigger event_flag=True (pre-release)."""
    import prism.news.intelligence as ni_mod
    from datetime import datetime, timezone, timedelta

    event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
    fake_events = [{
        "impact": "High",
        "currency": "USD",
        "date": event_time.isoformat(),
        "title": "NFP",
    }]

    class FakeResp:
        status_code = 200
        def json(self):
            return fake_events

    with patch.object(ni_mod.requests, "get", return_value=FakeResp()):
        ni = NewsIntelligence()
        flag, name = ni._check_economic_calendar("XAUUSD")
    assert flag is True
    assert name == "NFP"


def test_economic_calendar_blocks_just_after_event():
    """Within T+15m after a high-impact event, blackout must still be active."""
    import prism.news.intelligence as ni_mod
    from datetime import datetime, timezone, timedelta

    event_time = datetime.now(timezone.utc) - timedelta(minutes=5)  # 5 min ago
    fake_events = [{
        "impact": "High",
        "currency": "USD",
        "date": event_time.isoformat(),
        "title": "CPI",
    }]

    class FakeResp:
        status_code = 200
        def json(self):
            return fake_events

    with patch.object(ni_mod.requests, "get", return_value=FakeResp()):
        ni = NewsIntelligence()
        flag, _ = ni._check_economic_calendar("XAUUSD")
    assert flag is True


def test_economic_calendar_does_not_block_far_future_event():
    """Event 2 hours away must not trigger the blackout."""
    import prism.news.intelligence as ni_mod
    from datetime import datetime, timezone, timedelta

    event_time = datetime.now(timezone.utc) + timedelta(hours=2)
    fake_events = [{
        "impact": "High",
        "currency": "USD",
        "date": event_time.isoformat(),
        "title": "FOMC",
    }]

    class FakeResp:
        status_code = 200
        def json(self):
            return fake_events

    with patch.object(ni_mod.requests, "get", return_value=FakeResp()):
        ni = NewsIntelligence()
        flag, _ = ni._check_economic_calendar("XAUUSD")
    assert flag is False


# ---------------------------------------------------------------------------
# New: FVG retest confirmation + per-instrument persistence
# ---------------------------------------------------------------------------

def test_fvg_retest_requires_price_left_zone():
    """
    With M5 df where price never left the zone, retest_confirmed should fail
    and check_entry_trigger returns None.
    """
    detector = FVGDetector("EURUSD", "H4")
    detector.zones = [
        FVGZone("EURUSD", "H4", "BULLISH", 1.1050, 1.1000, 1.1025,
                "2024-01-01", 0, age_bars=5, strength=0.5),
    ]
    # M5 bars all sitting inside the zone — no break out, no retest
    m5 = pd.DataFrame([
        {"open": 1.1020, "high": 1.1030, "low": 1.1015, "close": 1.1025},
        {"open": 1.1025, "high": 1.1035, "low": 1.1020, "close": 1.1030},
        {"open": 1.1030, "high": 1.1040, "low": 1.1025, "close": 1.1030},
    ])
    assert detector.check_entry_trigger(1.1030, "LONG", m5_df=m5) is None


def test_fvg_retest_confirmed_after_breakout():
    """
    With M5 df where price impulsed above the zone then came back,
    check_entry_trigger should return the zone and mark retest_confirmed=True.
    """
    detector = FVGDetector("EURUSD", "H4")
    detector.zones = [
        FVGZone("EURUSD", "H4", "BULLISH", 1.1050, 1.1000, 1.1025,
                "2024-01-01", 0, age_bars=5, strength=0.5),
    ]
    # Prior bars: price well above zone (low > top). Current bar: back inside.
    m5 = pd.DataFrame([
        {"open": 1.1080, "high": 1.1100, "low": 1.1070, "close": 1.1090},
        {"open": 1.1085, "high": 1.1095, "low": 1.1065, "close": 1.1075},
        {"open": 1.1070, "high": 1.1075, "low": 1.1060, "close": 1.1065},
        {"open": 1.1040, "high": 1.1045, "low": 1.1025, "close": 1.1030},
    ])
    zone = detector.check_entry_trigger(1.1030, "LONG", m5_df=m5)
    assert zone is not None
    assert zone.retest_confirmed is True


def test_fvg_save_and_load_namespaced(tmp_path, monkeypatch):
    """Save writes to signals/fvg_zones_<INSTR>_<TF>.json; load round-trips."""
    import prism.signal.fvg as fvg_mod
    monkeypatch.setattr(fvg_mod, "FVG_STORE_DIR", tmp_path)
    detector = FVGDetector("EURUSD", "H4")
    detector.zones = [
        FVGZone("EURUSD", "H4", "BULLISH", 1.1050, 1.1000, 1.1025,
                "2024-01-01", 0, age_bars=5, strength=0.5),
    ]
    detector.save()
    expected = tmp_path / "fvg_zones_EURUSD_H4.json"
    assert expected.exists()

    loaded = FVGDetector.load("EURUSD", "H4")
    assert len(loaded.zones) == 1
    assert loaded.zones[0].bottom == pytest.approx(1.1000)


# ---------------------------------------------------------------------------
# New: MT5 Bridge CONFIRM mode returns PENDING_APPROVAL (no execution)
# ---------------------------------------------------------------------------

def test_mt5_confirm_mode_returns_pending_approval():
    """CONFIRM mode must not execute — caller must submit_order after approval."""
    from prism.execution.mt5_bridge import MT5Bridge
    bridge = MT5Bridge(mode="CONFIRM")
    # Do NOT connect — CONFIRM must gate before touching MT5 at all.
    signal = _make_signal_packet()
    result = bridge.execute_signal(signal)
    assert result.success is False
    assert result.status == "PENDING_APPROVAL"
    assert result.ticket is None
    assert result.error is None


def test_mt5_notify_mode_does_not_execute():
    from prism.execution.mt5_bridge import MT5Bridge
    bridge = MT5Bridge(mode="NOTIFY")
    signal = _make_signal_packet()
    result = bridge.execute_signal(signal)
    assert result.status == "NOTIFY"
    assert result.success is False


# ---------------------------------------------------------------------------
# New: MT5 filling-mode picker adapts to broker
# ---------------------------------------------------------------------------

def test_pick_filling_mode_prefers_ioc_when_supported():
    """symbol_info.filling_mode with bit 1 set → IOC."""
    from prism.execution.mt5_bridge import MT5Bridge
    bridge = MT5Bridge(mode="AUTO")
    mt5_mock = MagicMock()
    mt5_mock.ORDER_FILLING_IOC = "IOC"
    mt5_mock.ORDER_FILLING_FOK = "FOK"
    mt5_mock.ORDER_FILLING_RETURN = "RETURN"
    mt5_mock.symbol_info.return_value = MagicMock(filling_mode=2)  # IOC bit
    bridge._mt5 = mt5_mock
    assert bridge._pick_filling_mode("EURUSDm") == "IOC"


def test_pick_filling_mode_falls_back_to_return():
    from prism.execution.mt5_bridge import MT5Bridge
    bridge = MT5Bridge(mode="AUTO")
    mt5_mock = MagicMock()
    mt5_mock.ORDER_FILLING_IOC = "IOC"
    mt5_mock.ORDER_FILLING_FOK = "FOK"
    mt5_mock.ORDER_FILLING_RETURN = "RETURN"
    mt5_mock.symbol_info.return_value = MagicMock(filling_mode=4)  # only RETURN bit
    bridge._mt5 = mt5_mock
    assert bridge._pick_filling_mode("EURUSD") == "RETURN"


# ---------------------------------------------------------------------------
# New: End-to-end SignalGenerator → SignalPacket integration
# ---------------------------------------------------------------------------

def _build_h4_with_bullish_fvg(n: int = 40, base: float = 1.1000) -> pd.DataFrame:
    """
    Build an H4 frame ending in an unmitigated BULLISH FVG zone near the current price.
    Layout: flat rails, a single 3-bar FVG in the middle, then flat rails hovering
    inside the zone so the last bar's close sits between zone midline and top.
    """
    import datetime as _dt
    rows = []
    ts = datetime(2024, 1, 1)

    for i in range(n - 10):
        rows.append({
            "datetime": ts + _dt.timedelta(hours=4 * i),
            "open": base, "high": base + 0.0005, "low": base - 0.0005, "close": base,
            "volume": 100, "atr_14": 0.001,
        })

    # 3-bar bullish FVG: high of prev2 (1.1000) < low of curr (1.1030) → gap
    fvg_base_idx = len(rows)
    rows[-1]["high"] = 1.1000
    rows[-1]["low"] = 1.0990
    rows[-1]["close"] = 1.1000
    rows.append({
        "datetime": ts + _dt.timedelta(hours=4 * (len(rows))),
        "open": 1.1010, "high": 1.1040, "low": 1.1005, "close": 1.1035,
        "volume": 200, "atr_14": 0.001,
    })
    rows.append({
        "datetime": ts + _dt.timedelta(hours=4 * (len(rows))),
        "open": 1.1035, "high": 1.1060, "low": 1.1030, "close": 1.1050,
        "volume": 200, "atr_14": 0.001,
    })

    # Subsequent bars: dip into zone without closing below bottom (1.1000)
    for j in range(8):
        rows.append({
            "datetime": ts + _dt.timedelta(hours=4 * (len(rows))),
            "open": 1.1020, "high": 1.1045, "low": 1.1016, "close": 1.1020,
            "volume": 150, "atr_14": 0.001,
        })

    df = pd.DataFrame(rows)
    df["rsi_14"] = 55.0  # dummy feature so feature_cols is non-empty
    return df


def test_signal_generator_end_to_end_returns_packet(tmp_path, monkeypatch):
    """
    Integration: with mocked news + predictor + ICC + synthetic H4/M5 data aligning
    on a bullish FVG zone, generate() should produce a SignalPacket.
    """
    import prism.signal.fvg as fvg_mod
    from prism.signal.generator import SignalGenerator
    monkeypatch.setattr(fvg_mod, "FVG_STORE_DIR", tmp_path)

    gen = SignalGenerator("EURUSD")

    # Layer 0: clear
    gen.news.get_signal = MagicMock(return_value=NewsSignal(
        instrument="EURUSD",
        timestamp="2024-01-01T12:00:00",
        news_bias="BULLISH",
        event_flag=False,
        event_name="",
        risk_regime="NEUTRAL",
        sentiment_score=0.3,
        geopolitical_active=False,
        sources=[],
    ))
    gen.news.should_block_trade = MagicMock(return_value=(False, ""))

    # Layer 1: predictor says LONG high-conf
    mock_predictor = MagicMock()
    mock_predictor.predict_latest.return_value = {
        "direction": 1,
        "direction_str": "LONG",
        "confidence": 0.78,
        "confidence_level": "HIGH",
        "magnitude_pips": 35.0,
    }
    gen._predictor = mock_predictor

    # Layer 2: ICC continuation LONG
    gen.icc.detect_signals = MagicMock(return_value=[{
        "phase": "CONTINUATION",
        "direction": "LONG",
        "entry": 1.1020,
        "sl": 1.0990,
        "correction_low": 1.0995,
        "leg_size": 0.0060,
        "correction_pct": 0.4,
        "indication_range": 0.0050,
        "indication_range_pips": 50.0,
        "aoi_nearby": False,
    }])

    # Layer 3: build H4 dataframe with a real bullish FVG zone
    h4_df = _build_h4_with_bullish_fvg()

    # M5 df: prior bars sit above the zone top (1.1030) so retest fires, final bar inside.
    m5_rows = []
    for _ in range(3):
        m5_rows.append({"open": 1.1055, "high": 1.1065, "low": 1.1045, "close": 1.1060})
    m5_rows.append({"open": 1.1035, "high": 1.1040, "low": 1.1025, "close": 1.1030})
    entry_df = pd.DataFrame(m5_rows)
    entry_df["atr_14"] = 0.001

    packet = gen.generate(h4_df=h4_df, h1_df=h4_df, entry_df=entry_df)

    assert packet is not None, "Expected SignalPacket when all layers align"
    assert packet.direction == "LONG"
    assert packet.instrument == "EURUSD"
    assert packet.rr_ratio >= 1.5
    assert packet.sl < packet.entry < packet.tp2
    assert packet.fvg_zone is not None
    # Persistence side effect: per-instrument FVG store was written
    assert (tmp_path / "fvg_zones_EURUSD_H4.json").exists()


def test_signal_generator_skips_when_icc_disagrees_with_ml():
    """ML says LONG but ICC continuation is SHORT → generator must return None."""
    from prism.signal.generator import SignalGenerator
    gen = SignalGenerator("EURUSD", persist_fvg=False)

    gen.news.get_signal = MagicMock(return_value=NewsSignal(
        instrument="EURUSD", timestamp="t", news_bias="NEUTRAL",
        event_flag=False, event_name="", risk_regime="NEUTRAL",
        sentiment_score=0.0, geopolitical_active=False, sources=[],
    ))
    gen.news.should_block_trade = MagicMock(return_value=(False, ""))

    mock_predictor = MagicMock()
    mock_predictor.predict_latest.return_value = {
        "direction": 1, "direction_str": "LONG", "confidence": 0.75,
        "confidence_level": "HIGH", "magnitude_pips": 20.0,
    }
    gen._predictor = mock_predictor
    gen.icc.detect_signals = MagicMock(return_value=[{
        "phase": "CONTINUATION", "direction": "SHORT",
        "entry": 1.0, "sl": 1.1, "correction_high": 1.05,
        "leg_size": 0.005, "correction_pct": 0.4,
        "indication_range": 0.01, "indication_range_pips": 100.0,
        "aoi_nearby": False,
    }])

    df = _make_flat_df(30)
    df["rsi_14"] = 50.0
    assert gen.generate(h4_df=df, h1_df=df, entry_df=df) is None
