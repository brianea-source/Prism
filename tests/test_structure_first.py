"""Tests for the structure-first signal generator flow.

The key behavioral difference from legacy: market structure (HTF bias + ICC)
determines trade direction. ML provides confidence scoring. News bias adjusts
confidence but does not veto trades.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ohlcv(n: int = 50, base: float = 2000.0, trend: float = 0.0) -> pd.DataFrame:
    """Minimal OHLCV dataframe with enough bars for swing detection."""
    rows = []
    for i in range(n):
        c = base + trend * i
        rows.append({
            "datetime": datetime(2026, 5, 1, tzinfo=timezone.utc),
            "open": c, "high": c + 1, "low": c - 1, "close": c,
            "volume": 100,
        })
    return pd.DataFrame(rows)


def _h4_with_features(n: int = 50, base: float = 2000.0) -> pd.DataFrame:
    df = _ohlcv(n, base)
    df["feat_1"] = np.random.randn(n)
    df["feat_2"] = np.random.randn(n)
    df["atr_14"] = 5.0
    return df


@dataclass
class _FakeHTFResult:
    bias_1h: MagicMock
    bias_4h: MagicMock
    bias_daily = None
    swing_points_1h: list = None
    swing_points_4h: list = None
    aligned: bool = True
    allowed_direction: str = "LONG"

    def __post_init__(self):
        if self.swing_points_1h is None:
            self.swing_points_1h = [{"type": "HH"}, {"type": "HL"}]
        if self.swing_points_4h is None:
            self.swing_points_4h = [{"type": "HH"}, {"type": "HL"}]


def _bullish_htf():
    b = MagicMock()
    b.value = "BULLISH"
    return _FakeHTFResult(bias_1h=b, bias_4h=b, aligned=True, allowed_direction="LONG")


def _bearish_htf():
    b = MagicMock()
    b.value = "BEARISH"
    return _FakeHTFResult(bias_1h=b, bias_4h=b, aligned=True, allowed_direction="SHORT")


def _ranging_htf():
    b = MagicMock()
    b.value = "RANGING"
    return _FakeHTFResult(bias_1h=b, bias_4h=b, aligned=False, allowed_direction=None)


def _icc_continuation(direction: str = "LONG"):
    return [{
        "phase": "CONTINUATION",
        "direction": direction,
        "entry": 2000.0,
        "sl": 1990.0,
        "correction_low": 1992.0,
        "correction_high": 2008.0,
        "correction_pct": 0.4,
        "indication_range": 20.0,
        "indication_range_pips": 200.0,
        "leg_size": 12.36,
    }]


def _ml_prediction(direction_str="SHORT", confidence=0.72):
    return {
        "direction": 1 if direction_str == "LONG" else -1,
        "direction_str": direction_str,
        "confidence": confidence,
        "confidence_level": "HIGH" if confidence > 0.7 else "MEDIUM",
        "magnitude_pips": 25.0,
    }


def _news_neutral():
    return MagicMock(
        news_bias="NEUTRAL",
        risk_regime="NEUTRAL",
        event_flag=False,
    )


def _news_bullish():
    return MagicMock(
        news_bias="BULLISH",
        risk_regime="NEUTRAL",
        event_flag=False,
    )


def _news_bearish():
    return MagicMock(
        news_bias="BEARISH",
        risk_regime="NEUTRAL",
        event_flag=False,
    )


def _fvg_zone():
    return MagicMock(
        top=2005.0, bottom=1995.0, midline=2000.0,
        direction="LONG", state="ACTIVE",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _structure_first_on(monkeypatch):
    """Ensure structure-first is ON for all tests in this file."""
    monkeypatch.setenv("PRISM_STRUCTURE_FIRST", "1")
    monkeypatch.setenv("PRISM_SMART_MONEY_ENABLED", "0")
    monkeypatch.setenv("PRISM_HTF_ENABLED", "1")


# ---------------------------------------------------------------------------
# Tests: structure determines direction, ML does not
# ---------------------------------------------------------------------------
class TestStructureFirstDirection:
    """The critical scenario: HTF says LONG, ML says SHORT, trade goes LONG."""

    def test_htf_long_ml_short_fires_long(self, monkeypatch):
        """When HTF + ICC say LONG and ML says SHORT, the signal should
        fire LONG — structure is the authority on direction."""
        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)

        # Patch news
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_neutral()
        gen.news.should_block_trade.return_value = (False, "")

        # Patch HTF → BULLISH
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = _bullish_htf()

        # Patch ICC → LONG continuation
        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = _icc_continuation("LONG")

        # Patch ML → SHORT (disagrees with structure!)
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = _ml_prediction("SHORT", 0.72)

        # Patch FVG → triggers
        gen.fvg = MagicMock()
        gen.fvg.check_entry_trigger.return_value = _fvg_zone()

        h4 = _h4_with_features()
        h1 = _ohlcv()
        entry = _ohlcv()

        packet = gen.generate(h4, h1, entry)

        assert packet is not None, "Signal should fire despite ML disagreeing"
        assert packet.direction == "LONG", "Direction should follow structure, not ML"

    def test_htf_short_ml_long_fires_short(self, monkeypatch):
        """Mirror case: HTF says SHORT, ML says LONG, should fire SHORT."""
        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)

        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_neutral()
        gen.news.should_block_trade.return_value = (False, "")

        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = _bearish_htf()

        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = _icc_continuation("SHORT")

        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = _ml_prediction("LONG", 0.72)

        gen.fvg = MagicMock()
        gen.fvg.check_entry_trigger.return_value = _fvg_zone()

        h4 = _h4_with_features()
        packet = gen.generate(h4, _ohlcv(), _ohlcv())

        assert packet is not None
        assert packet.direction == "SHORT"


# ---------------------------------------------------------------------------
# Tests: HTF alignment gate
# ---------------------------------------------------------------------------
class TestHTFGate:
    def test_ranging_htf_blocks(self, monkeypatch):
        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_neutral()
        gen.news.should_block_trade.return_value = (False, "")
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = _ranging_htf()

        packet = gen.generate(_h4_with_features(), _ohlcv(), _ohlcv())

        assert packet is None
        assert gen.last_rejection_gate == "htf_bias"


# ---------------------------------------------------------------------------
# Tests: ICC direction must agree with HTF
# ---------------------------------------------------------------------------
class TestICCHTFAgreement:
    def test_icc_disagrees_with_htf_blocks(self, monkeypatch):
        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_neutral()
        gen.news.should_block_trade.return_value = (False, "")
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = _bullish_htf()
        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = _icc_continuation("SHORT")

        packet = gen.generate(_h4_with_features(), _ohlcv(), _ohlcv())

        assert packet is None
        assert gen.last_rejection_gate == "icc_direction"


# ---------------------------------------------------------------------------
# Tests: news bias is soft penalty, not veto
# ---------------------------------------------------------------------------
class TestNewsBiasPenalty:
    def test_opposing_news_reduces_confidence_not_veto(self, monkeypatch):
        """News says BEARISH, structure says LONG — trade should still fire
        but with reduced confidence."""
        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_bearish()
        gen.news.should_block_trade.return_value = (False, "")
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = _bullish_htf()
        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = _icc_continuation("LONG")
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = _ml_prediction("LONG", 0.72)
        gen.fvg = MagicMock()
        gen.fvg.check_entry_trigger.return_value = _fvg_zone()

        packet = gen.generate(_h4_with_features(), _ohlcv(), _ohlcv())

        assert packet is not None, "News should not veto when confidence stays above threshold"
        assert packet.confidence == pytest.approx(0.62, abs=0.001), (
            "Confidence 0.72 minus 0.10 news penalty should equal 0.62"
        )

    def test_opposing_news_kills_low_confidence(self, monkeypatch):
        """If news penalty drops confidence below threshold, trade is skipped."""
        monkeypatch.setenv("PRISM_MIN_CONFIDENCE", "0.60")
        monkeypatch.setenv("PRISM_NEWS_BIAS_PENALTY", "0.10")

        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_bearish()
        gen.news.should_block_trade.return_value = (False, "")
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = _bullish_htf()
        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = _icc_continuation("LONG")
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = _ml_prediction("LONG", 0.62)
        gen.fvg = MagicMock()
        gen.fvg.check_entry_trigger.return_value = _fvg_zone()

        packet = gen.generate(_h4_with_features(), _ohlcv(), _ohlcv())

        assert packet is None
        assert gen.last_rejection_gate == "news_bias_penalty"

    def test_agreeing_news_no_penalty(self, monkeypatch):
        """News agrees with structure — no confidence penalty."""
        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_bullish()
        gen.news.should_block_trade.return_value = (False, "")
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = _bullish_htf()
        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = _icc_continuation("LONG")
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = _ml_prediction("LONG", 0.72)
        gen.fvg = MagicMock()
        gen.fvg.check_entry_trigger.return_value = _fvg_zone()

        packet = gen.generate(_h4_with_features(), _ohlcv(), _ohlcv())

        assert packet is not None
        assert packet.confidence == 0.72


# ---------------------------------------------------------------------------
# Tests: ML confidence threshold
# ---------------------------------------------------------------------------
class TestMLConfidence:
    def test_low_ml_confidence_blocks(self, monkeypatch):
        monkeypatch.setenv("PRISM_MIN_CONFIDENCE", "0.60")

        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_neutral()
        gen.news.should_block_trade.return_value = (False, "")
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = _bullish_htf()
        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = _icc_continuation("LONG")
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = _ml_prediction("LONG", 0.45)

        packet = gen.generate(_h4_with_features(), _ohlcv(), _ohlcv())

        assert packet is None
        assert gen.last_rejection_gate == "ml_confidence"


# ---------------------------------------------------------------------------
# Tests: legacy flow preserved
# ---------------------------------------------------------------------------
class TestLegacyFlow:
    def test_legacy_flow_rejects_ml_news_conflict(self, monkeypatch):
        """With PRISM_STRUCTURE_FIRST=0, the old ML-direction-first behavior
        is preserved: ML SHORT + news BULLISH = skip."""
        monkeypatch.setenv("PRISM_STRUCTURE_FIRST", "0")

        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_bullish()
        gen.news.should_block_trade.return_value = (False, "")
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = _ml_prediction("SHORT", 0.72)

        packet = gen.generate(_h4_with_features(), _ohlcv(), _ohlcv())

        assert packet is None
        assert gen.last_rejection_gate == "news_bias"


# ---------------------------------------------------------------------------
# Tests: last_rejection_gate tracking
# ---------------------------------------------------------------------------
class TestRejectionGate:
    def test_news_blackout_tagged(self, monkeypatch):
        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_neutral()
        gen.news.should_block_trade.return_value = (True, "NFP in 15 min")

        packet = gen.generate(_h4_with_features(), _ohlcv(), _ohlcv())

        assert packet is None
        assert gen.last_rejection_gate == "news_blackout"

    def test_fvg_entry_tagged(self, monkeypatch):
        from prism.signal.generator import SignalGenerator

        gen = SignalGenerator("XAUUSD", persist_fvg=False)
        gen.news = MagicMock()
        gen.news.get_signal.return_value = _news_neutral()
        gen.news.should_block_trade.return_value = (False, "")
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = _bullish_htf()
        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = _icc_continuation("LONG")
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = _ml_prediction("LONG", 0.72)
        gen.fvg = MagicMock()
        gen.fvg.check_entry_trigger.return_value = None  # No FVG zone

        packet = gen.generate(_h4_with_features(), _ohlcv(), _ohlcv())

        assert packet is None
        assert gen.last_rejection_gate == "fvg_entry"
