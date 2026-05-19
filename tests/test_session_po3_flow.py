"""End-to-end tests for the session Po3 generator flow.

These tests exercise ``_generate_session_po3`` through the public
``SignalGenerator.generate()`` interface with ``PRISM_SESSION_PO3=1``.

Each test mocks the external dependencies (news, ML predictor, FVG, MT5)
and verifies that the session-level direction from the Asian sweep overrides
ML direction, HTF adjusts confidence, and the full pipeline works.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from prism.signal.session_bias import AsianRange, SessionPhase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _m5_bar(dt, o, h, l, c):
    return {"dt": dt, "open": o, "high": h, "low": l, "close": c}


def _build_m5_df(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    df.index = pd.DatetimeIndex(df.pop("dt"))
    return df


def _build_h4_df(n: int = 50, base: float = 3200.0) -> pd.DataFrame:
    """H4 DataFrame with synthetic features (for ML)."""
    dates = pd.date_range("2026-05-10", periods=n, freq="4h", tz="UTC")
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "datetime": dates,
        "open": base + rng.randn(n) * 5,
        "high": base + 10 + rng.randn(n) * 5,
        "low": base - 10 + rng.randn(n) * 5,
        "close": base + rng.randn(n) * 5,
        "volume": rng.randint(1000, 5000, n),
        "sma_20": base + rng.randn(n),
        "ema_50": base + rng.randn(n),
        "rsi_14": 50 + rng.randn(n) * 10,
        "atr_14": 15.0 + rng.randn(n),
    }, index=dates)
    return df


def _build_h1_df(n: int = 100, base: float = 3200.0) -> pd.DataFrame:
    dates = pd.date_range("2026-05-15", periods=n, freq="1h", tz="UTC")
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "open": base + rng.randn(n) * 5,
        "high": base + 10 + rng.randn(n) * 5,
        "low": base - 10 + rng.randn(n) * 5,
        "close": base + rng.randn(n) * 5,
    }, index=dates)


def _full_m5_with_sweep(sweep_side: str = "LOW") -> pd.DataFrame:
    """Build a full M5 dataset with Asian range + London sweep + displacement.

    Asian range: 3200–3220 (00:00–06:00 UTC on 2026-05-19)
    """
    ref = date(2026, 5, 19)
    bars = []

    # Asian session: 72 bars of 3200–3220 range
    for i in range(72):
        minute = i * 5
        hour, mins = divmod(minute, 60)
        dt = datetime(ref.year, ref.month, ref.day, hour, mins, tzinfo=timezone.utc)
        mid = 3210.0
        price = mid + 10 * np.sin(2 * np.pi * i / 72)
        bars.append(_m5_bar(dt, price - 0.5, price + 1.0, price - 1.0, price + 0.3))

    if sweep_side == "LOW":
        # London sweep low: wick below 3200, close back above
        bars.append(_m5_bar(
            datetime(2026, 5, 19, 7, 30, tzinfo=timezone.utc),
            3203.0, 3205.0, 3195.0, 3204.0,
        ))
        # Displacement: price moves up 20 pips from sweep
        bars.append(_m5_bar(
            datetime(2026, 5, 19, 7, 35, tzinfo=timezone.utc),
            3204.0, 3216.0, 3203.0, 3215.0,
        ))
        # More London bars trending up
        for i in range(20):
            total_min = 7 * 60 + 40 + i * 5
            h, m = divmod(total_min, 60)
            dt = datetime(2026, 5, 19, h, m, tzinfo=timezone.utc)
            base = 3215.0 + i * 0.5
            bars.append(_m5_bar(dt, base, base + 2, base - 1, base + 1))
    else:
        # London sweep high: wick above 3220, close back below
        bars.append(_m5_bar(
            datetime(2026, 5, 19, 7, 30, tzinfo=timezone.utc),
            3218.0, 3225.0, 3215.0, 3216.0,
        ))
        # Displacement: price drops 20 pips
        bars.append(_m5_bar(
            datetime(2026, 5, 19, 7, 35, tzinfo=timezone.utc),
            3216.0, 3217.0, 3204.0, 3205.0,
        ))
        for i in range(20):
            total_min = 7 * 60 + 40 + i * 5
            h, m = divmod(total_min, 60)
            dt = datetime(2026, 5, 19, h, m, tzinfo=timezone.utc)
            base = 3205.0 - i * 0.5
            bars.append(_m5_bar(dt, base, base + 1, base - 2, base - 1))

    return _build_m5_df(bars)


# ---------------------------------------------------------------------------
# Mock factory
# ---------------------------------------------------------------------------

def _mock_news(event_flag=False, event_name="", news_bias="NEUTRAL",
               geopolitical_active=False, risk_regime="NEUTRAL"):
    ns = MagicMock()
    ns.event_flag = event_flag
    ns.event_name = event_name
    ns.news_bias = news_bias
    ns.geopolitical_active = geopolitical_active
    ns.risk_regime = risk_regime
    ns.sentiment_score = 0.0
    ns.sources = ["mock"]
    ns.volume_anomaly = False
    return ns


def _mock_predictor(direction_str="LONG", confidence=0.72, magnitude=30.0):
    pred = MagicMock()
    pred.predict_latest.return_value = {
        "direction": 1 if direction_str == "LONG" else -1,
        "direction_str": direction_str,
        "confidence": confidence,
        "confidence_level": "HIGH" if confidence > 0.65 else "MEDIUM",
        "magnitude_pips": magnitude,
    }
    return pred


def _mock_fvg(direction: str):
    """Return a mock FVG zone that triggers for the given direction."""
    zone = MagicMock()
    zone.midline = 3210.0
    zone.top = 3215.0
    zone.bottom = 3205.0
    zone.direction = direction
    zone.strength = 1.5
    zone.age_bars = 3
    zone.mitigated = False
    return zone


# ---------------------------------------------------------------------------
# Core flow tests
# ---------------------------------------------------------------------------

class TestSessionPo3Flow:
    """Tests that exercise the full _generate_session_po3 flow."""

    @pytest.fixture(autouse=True)
    def _enable_session_po3(self, monkeypatch):
        monkeypatch.setenv("PRISM_SESSION_PO3", "1")
        monkeypatch.setenv("PRISM_SESSION_DISPLACEMENT_PIPS", "10")
        monkeypatch.setenv("PRISM_SMART_MONEY_ENABLED", "0")

    def _make_generator(self, monkeypatch, news=None, predictor=None,
                        fvg_zone=None, sweep_recent=True):
        from prism.signal.generator import SignalGenerator
        gen = SignalGenerator("XAUUSD")
        gen.news = MagicMock()
        gen.news.get_signal.return_value = news or _mock_news()
        gen.news.should_block_trade.return_value = (False, "")
        gen._predictor = predictor or _mock_predictor()
        gen.fvg = MagicMock()
        gen.fvg.check_entry_trigger.return_value = fvg_zone
        gen.sweep_detector = MagicMock()
        gen.sweep_detector.detect.return_value = []
        gen.sweep_detector.has_recent_sweep.return_value = sweep_recent
        return gen

    def test_low_sweep_fires_long(self, monkeypatch):
        """Asian LOW swept → LONG signal fires (the May 4-18 scenario fixed)."""
        entry_df = _full_m5_with_sweep("LOW")
        gen = self._make_generator(
            monkeypatch,
            predictor=_mock_predictor("SHORT", 0.72),
            fvg_zone=_mock_fvg("LONG"),
        )

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        assert signal is not None
        assert signal.direction == "LONG"

    def test_high_sweep_fires_short(self, monkeypatch):
        """Asian HIGH swept → SHORT signal fires."""
        entry_df = _full_m5_with_sweep("HIGH")
        gen = self._make_generator(
            monkeypatch,
            predictor=_mock_predictor("LONG", 0.72),
            fvg_zone=_mock_fvg("SHORT"),
        )

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        assert signal is not None
        assert signal.direction == "SHORT"

    def test_ml_direction_ignored_for_trade_direction(self, monkeypatch):
        """ML says SHORT but session Po3 says LONG → fires LONG."""
        entry_df = _full_m5_with_sweep("LOW")
        gen = self._make_generator(
            monkeypatch,
            predictor=_mock_predictor("SHORT", 0.72),
            fvg_zone=_mock_fvg("LONG"),
        )

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        assert signal is not None
        assert signal.direction == "LONG"  # session Po3, not ML

    def test_fomc_day_blocks_via_session_quality(self, monkeypatch):
        """FOMC day = SKIP grade → no signal."""
        entry_df = _full_m5_with_sweep("LOW")
        gen = self._make_generator(
            monkeypatch,
            news=_mock_news(event_flag=True, event_name="FOMC Federal Funds Rate"),
            fvg_zone=_mock_fvg("LONG"),
        )

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        assert signal is None
        assert gen.last_rejection_gate == "session_quality"

    def test_no_sweep_yet_blocks(self, monkeypatch):
        """No Asian sweep detected → no signal (waiting for manipulation)."""
        # Build M5 data without a sweep (all prices inside Asian range)
        ref = date(2026, 5, 19)
        bars = []
        for i in range(72):
            minute = i * 5
            h, m = divmod(minute, 60)
            dt = datetime(ref.year, ref.month, ref.day, h, m, tzinfo=timezone.utc)
            bars.append(_m5_bar(dt, 3210.0, 3215.0, 3205.0, 3212.0))
        # London bars inside the Asian range (no sweep — highs/lows stay within)
        for i in range(10):
            dt = datetime(2026, 5, 19, 7, i * 5, tzinfo=timezone.utc)
            bars.append(_m5_bar(dt, 3210.0, 3214.0, 3206.0, 3211.0))
        entry_df = _build_m5_df(bars)

        gen = self._make_generator(monkeypatch, fvg_zone=_mock_fvg("LONG"))
        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        assert signal is None
        assert gen.last_rejection_gate == "session_po3_no_sweep"

    def test_no_fvg_blocks(self, monkeypatch):
        """Sweep + displacement OK, but no FVG entry → blocked."""
        entry_df = _full_m5_with_sweep("LOW")
        gen = self._make_generator(
            monkeypatch,
            fvg_zone=None,  # no FVG
        )

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        assert signal is None
        assert gen.last_rejection_gate == "fvg_entry"

    def test_no_entry_tf_sweep_blocks(self, monkeypatch):
        """Session sweep OK, but no confirming sweep on entry TF → blocked."""
        entry_df = _full_m5_with_sweep("LOW")
        gen = self._make_generator(
            monkeypatch,
            fvg_zone=_mock_fvg("LONG"),
            sweep_recent=False,
        )

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        assert signal is None
        assert gen.last_rejection_gate == "sweep_confirmation"

    def test_low_ml_confidence_blocks(self, monkeypatch):
        """ML confidence too low + session blending → below threshold."""
        monkeypatch.setenv("PRISM_MIN_CONFIDENCE", "0.65")
        entry_df = _full_m5_with_sweep("LOW")
        gen = self._make_generator(
            monkeypatch,
            predictor=_mock_predictor("LONG", 0.40),
            fvg_zone=_mock_fvg("LONG"),
        )

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        # Blended: (0.70 + 0.40) / 2 = 0.55 < 0.65
        assert signal is None
        assert gen.last_rejection_gate == "ml_confidence"

    def test_opposing_news_reduces_confidence(self, monkeypatch):
        """Opposing news applies penalty but doesn't veto if confidence survives."""
        monkeypatch.setenv("PRISM_MIN_CONFIDENCE", "0.55")
        monkeypatch.setenv("PRISM_NEWS_BIAS_PENALTY", "0.05")
        entry_df = _full_m5_with_sweep("LOW")
        gen = self._make_generator(
            monkeypatch,
            news=_mock_news(news_bias="BEARISH"),
            predictor=_mock_predictor("LONG", 0.72),
            fvg_zone=_mock_fvg("LONG"),
        )

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        # Blended: (0.70 + 0.72) / 2 = 0.71 - 0.05 penalty = 0.66 > 0.55
        assert signal is not None
        assert signal.confidence < 0.71  # penalty applied

    def test_signal_packet_contains_session_po3_metadata(self, monkeypatch):
        """SignalPacket.smart_money has session_po3 and session_quality dicts."""
        entry_df = _full_m5_with_sweep("LOW")
        gen = self._make_generator(
            monkeypatch,
            predictor=_mock_predictor("LONG", 0.72),
            fvg_zone=_mock_fvg("LONG"),
        )

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        assert signal is not None
        assert "session_po3" in signal.smart_money
        assert signal.smart_money["session_po3"]["sweep_side"] == "LOW"
        assert signal.smart_money["session_po3"]["phase"] == "DISTRIBUTION"
        assert "session_quality" in signal.smart_money
        assert signal.smart_money["session_quality"]["grade"] == "FAVORABLE"

    def test_htf_agreement_boosts_confidence(self, monkeypatch):
        """HTF agrees with session direction → confidence gets a bonus."""
        monkeypatch.setenv("PRISM_HTF_AGREE_BONUS", "0.10")
        monkeypatch.setenv("PRISM_MIN_CONFIDENCE", "0.55")
        entry_df = _full_m5_with_sweep("LOW")

        gen = self._make_generator(
            monkeypatch,
            predictor=_mock_predictor("LONG", 0.50),
            fvg_zone=_mock_fvg("LONG"),
        )
        # Mock HTF to agree with LONG
        gen.htf_engine = MagicMock()
        htf_result = MagicMock()
        htf_result.aligned = True
        htf_result.allowed_direction = "LONG"
        htf_result.bias_1h = MagicMock(value="BULLISH")
        htf_result.bias_4h = MagicMock(value="BULLISH")
        gen.htf_engine.refresh.return_value = htf_result

        signal = gen.generate(_build_h4_df(), _build_h1_df(), entry_df)
        # Base 0.70 + HTF bonus 0.10 = 0.80; blend with ML 0.50 → 0.65
        # That's above 0.55, so it fires
        assert signal is not None
        assert signal.confidence >= 0.60


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_session_po3_takes_priority_over_structure_first(self, monkeypatch):
        """PRISM_SESSION_PO3=1 overrides PRISM_STRUCTURE_FIRST=1."""
        monkeypatch.setenv("PRISM_SESSION_PO3", "1")
        monkeypatch.setenv("PRISM_STRUCTURE_FIRST", "1")

        from prism.signal.generator import SignalGenerator
        gen = SignalGenerator("XAUUSD")

        # We just need to verify it dispatches to the right method
        gen._generate_session_po3 = MagicMock(return_value=None)
        gen._generate_structure_first = MagicMock(return_value=None)
        gen._generate_legacy = MagicMock(return_value=None)

        gen.generate(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        gen._generate_session_po3.assert_called_once()
        gen._generate_structure_first.assert_not_called()
        gen._generate_legacy.assert_not_called()

    def test_structure_first_used_when_session_po3_off(self, monkeypatch):
        monkeypatch.setenv("PRISM_SESSION_PO3", "0")
        monkeypatch.setenv("PRISM_STRUCTURE_FIRST", "1")

        from prism.signal.generator import SignalGenerator
        gen = SignalGenerator("XAUUSD")

        gen._generate_session_po3 = MagicMock(return_value=None)
        gen._generate_structure_first = MagicMock(return_value=None)

        gen.generate(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        gen._generate_session_po3.assert_not_called()
        gen._generate_structure_first.assert_called_once()
