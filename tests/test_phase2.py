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
