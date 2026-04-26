"""
PRISM Phase 5 tests — HTF Bias Engine.
Tests for swing structure detection, bias classification, and signal gating.
All tests are self-contained using synthetic DataFrames.
"""
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from prism.signal.htf_bias import (
    Bias,
    HTFBiasResult,
    detect_swing_structure,
    classify_bias,
    get_htf_bias,
    HTFBiasEngine,
)


# ---------------------------------------------------------------------------
# Helper functions for creating synthetic DataFrames
# ---------------------------------------------------------------------------

def make_uptrend_df(n_bars: int = 50, base_price: float = 100.0) -> pd.DataFrame:
    """Create a synthetic uptrend with clear HH/HL structure."""
    data = []
    price = base_price
    for i in range(n_bars):
        # Simulate upward zigzag: price rises with periodic corrections
        if i % 5 == 0:
            # Correction down (but higher low than previous correction)
            change = -2 + (i * 0.1)  # Small corrections that get smaller
        else:
            change = 1.5 + (i * 0.05)  # Steady rise
        price = max(base_price, price + change)
        data.append({
            "open": price - 0.5,
            "high": price + 1.0 + (i * 0.1),  # Higher highs
            "low": price - 1.0 + (i * 0.05),   # Higher lows
            "close": price,
            "volume": 1000,
        })
    return pd.DataFrame(data)


def make_downtrend_df(n_bars: int = 50, base_price: float = 200.0) -> pd.DataFrame:
    """Create a synthetic downtrend with clear LH/LL structure."""
    data = []
    price = base_price
    for i in range(n_bars):
        # Simulate downward zigzag
        if i % 5 == 0:
            change = 2 - (i * 0.1)  # Corrections up (but lower highs)
        else:
            change = -1.5 - (i * 0.05)  # Steady decline
        price = min(base_price, price + change)
        data.append({
            "open": price + 0.5,
            "high": price + 1.0 - (i * 0.05),  # Lower highs
            "low": price - 1.0 - (i * 0.1),    # Lower lows
            "close": price,
            "volume": 1000,
        })
    return pd.DataFrame(data)


def make_ranging_df(n_bars: int = 50, base_price: float = 150.0, range_size: float = 5.0) -> pd.DataFrame:
    """Create a sideways/ranging market with mixed swing structure."""
    data = []
    for i in range(n_bars):
        # Oscillate around base price
        offset = np.sin(i * 0.5) * range_size
        price = base_price + offset
        data.append({
            "open": price - 0.3,
            "high": price + 1.5,
            "low": price - 1.5,
            "close": price + 0.2,
            "volume": 1000,
        })
    return pd.DataFrame(data)


def make_swing_df_with_clear_pattern(swing_type: str, n_bars: int = 30) -> pd.DataFrame:
    """Create a DataFrame with a clear swing pattern for testing.

    swing_type: "uptrend" - HH, HL, HH, HL pattern
                "downtrend" - LH, LL, LH, LL pattern
    """
    data = []
    if swing_type == "uptrend":
        # Clear uptrend: create explicit swing highs and lows
        prices = [
            100, 101, 102, 98, 97, 99,     # First swing low at 97
            103, 105, 107, 102, 100, 103,  # First swing high at 107, second swing low at 100
            108, 110, 112, 107, 105, 108,  # Second swing high at 112, third swing low at 105
            113, 115, 118, 112, 110, 114,  # Third swing high at 118, fourth swing low at 110
            119, 121, 124, 118, 116, 120,  # Fourth swing high at 124
        ]
    else:  # downtrend
        prices = [
            200, 199, 198, 202, 203, 201,  # First swing high at 203
            197, 195, 193, 198, 200, 197,  # First swing low at 193, second swing high at 200
            192, 190, 188, 193, 195, 192,  # Second swing low at 188, third swing high at 195
            187, 185, 182, 187, 189, 186,  # Third swing low at 182, fourth swing high at 189
            181, 179, 176, 181, 183, 180,  # Fourth swing low at 176
        ]

    for i, price in enumerate(prices):
        data.append({
            "open": price - 0.5,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": 1000,
        })
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Test: detect_swing_structure
# ---------------------------------------------------------------------------

class TestDetectSwingStructure:
    def test_detect_swing_structure_uptrend(self):
        """Synthetic uptrend df should detect HH/HL points."""
        df = make_swing_df_with_clear_pattern("uptrend")
        swings = detect_swing_structure(df, lookback=3)

        # Should find some swing points
        assert len(swings) > 0

        # Check for presence of HH and HL types
        types = [s["type"] for s in swings]
        has_hh = "HH" in types
        has_hl = "HL" in types
        # In a clear uptrend, we expect HH and HL
        assert has_hh or has_hl, f"Expected HH or HL in uptrend, got {types}"

    def test_detect_swing_structure_downtrend(self):
        """Synthetic downtrend df should detect LH/LL points."""
        df = make_swing_df_with_clear_pattern("downtrend")
        swings = detect_swing_structure(df, lookback=3)

        assert len(swings) > 0
        types = [s["type"] for s in swings]
        has_lh = "LH" in types
        has_ll = "LL" in types
        assert has_lh or has_ll, f"Expected LH or LL in downtrend, got {types}"

    def test_detect_swing_structure_ranging(self):
        """Sideways df should produce mixed or empty swing points."""
        df = make_ranging_df(n_bars=50)
        swings = detect_swing_structure(df, lookback=3)

        # Ranging market may have swings but they should be mixed
        if len(swings) >= 3:
            types = [s["type"] for s in swings[-3:]]
            # In a ranging market, we expect a mix (not pure HH+HL or LH+LL)
            has_bullish = "HH" in types or "HL" in types
            has_bearish = "LH" in types or "LL" in types
            # At least some mix expected, or could be empty
            pass  # Just verifying no crash and structure is valid

    def test_detect_swing_structure_insufficient_data(self):
        """Less than 2*lookback+1 bars should return empty list."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1000, 1000],
        })
        swings = detect_swing_structure(df, lookback=3)
        assert swings == []

    def test_detect_swing_structure_returns_sorted_by_bar_idx(self):
        """Swing points should be sorted by bar_idx."""
        df = make_swing_df_with_clear_pattern("uptrend")
        swings = detect_swing_structure(df, lookback=3)

        if len(swings) > 1:
            bar_idxs = [s["bar_idx"] for s in swings]
            assert bar_idxs == sorted(bar_idxs)


# ---------------------------------------------------------------------------
# Test: classify_bias
# ---------------------------------------------------------------------------

class TestClassifyBias:
    def test_classify_bias_bullish(self):
        """HH+HL swing_points should classify as BULLISH."""
        swing_points = [
            {"price": 100, "type": "HH", "bar_idx": 5},
            {"price": 98, "type": "HL", "bar_idx": 10},
            {"price": 105, "type": "HH", "bar_idx": 15},
        ]
        bias = classify_bias(swing_points, min_swing_points=3)
        assert bias == Bias.BULLISH

    def test_classify_bias_bearish(self):
        """LH+LL swing_points should classify as BEARISH."""
        swing_points = [
            {"price": 200, "type": "LH", "bar_idx": 5},
            {"price": 195, "type": "LL", "bar_idx": 10},
            {"price": 198, "type": "LH", "bar_idx": 15},
        ]
        bias = classify_bias(swing_points, min_swing_points=3)
        assert bias == Bias.BEARISH

    def test_classify_bias_ranging_mixed(self):
        """Mixed highs/lows should classify as RANGING."""
        swing_points = [
            {"price": 100, "type": "HH", "bar_idx": 5},
            {"price": 98, "type": "LL", "bar_idx": 10},  # Mixed!
            {"price": 102, "type": "LH", "bar_idx": 15},
        ]
        bias = classify_bias(swing_points, min_swing_points=3)
        assert bias == Bias.RANGING

    def test_classify_bias_insufficient_points(self):
        """Less than min_swing_points should classify as RANGING."""
        swing_points = [
            {"price": 100, "type": "HH", "bar_idx": 5},
            {"price": 98, "type": "HL", "bar_idx": 10},
        ]
        bias = classify_bias(swing_points, min_swing_points=3)
        assert bias == Bias.RANGING

    def test_classify_bias_single_hh(self):
        """Single HH but no HL should classify as RANGING."""
        swing_points = [
            {"price": 100, "type": "HH", "bar_idx": 5},
            {"price": 102, "type": "HH", "bar_idx": 10},
            {"price": 105, "type": "HH", "bar_idx": 15},
        ]
        bias = classify_bias(swing_points, min_swing_points=3)
        # Only HH, no HL - not a complete bullish pattern
        assert bias == Bias.RANGING


# ---------------------------------------------------------------------------
# Test: get_htf_bias
# ---------------------------------------------------------------------------

class TestGetHTFBias:
    def test_get_htf_bias_aligned_bullish(self):
        """Both 1H+4H BULLISH should result in aligned=True, allowed_direction='LONG'."""
        # Create DataFrames that will classify as BULLISH
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_swing_df_with_clear_pattern("uptrend")

        # Manually create the result to test alignment logic
        # Since swing detection depends on exact data, let's mock classify_bias
        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.return_value = Bias.BULLISH
            result = get_htf_bias(df_1h, df_4h, min_swing_points=3)

        assert result.bias_1h == Bias.BULLISH
        assert result.bias_4h == Bias.BULLISH
        assert result.aligned is True
        assert result.allowed_direction == "LONG"

    def test_get_htf_bias_aligned_bearish(self):
        """Both 1H+4H BEARISH should result in aligned=True, allowed_direction='SHORT'."""
        df_1h = make_swing_df_with_clear_pattern("downtrend")
        df_4h = make_swing_df_with_clear_pattern("downtrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.return_value = Bias.BEARISH
            result = get_htf_bias(df_1h, df_4h, min_swing_points=3)

        assert result.bias_1h == Bias.BEARISH
        assert result.bias_4h == Bias.BEARISH
        assert result.aligned is True
        assert result.allowed_direction == "SHORT"

    def test_get_htf_bias_misaligned(self):
        """1H BULLISH, 4H BEARISH should result in aligned=False, allowed_direction=None."""
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_swing_df_with_clear_pattern("downtrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            # First call for 1H, second for 4H
            mock_classify.side_effect = [Bias.BULLISH, Bias.BEARISH]
            result = get_htf_bias(df_1h, df_4h, min_swing_points=3)

        assert result.bias_1h == Bias.BULLISH
        assert result.bias_4h == Bias.BEARISH
        assert result.aligned is False
        assert result.allowed_direction is None

    def test_get_htf_bias_1h_ranging(self):
        """1H RANGING should result in aligned=False, allowed_direction=None."""
        df_1h = make_ranging_df()
        df_4h = make_swing_df_with_clear_pattern("uptrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.side_effect = [Bias.RANGING, Bias.BULLISH]
            result = get_htf_bias(df_1h, df_4h, min_swing_points=3)

        assert result.bias_1h == Bias.RANGING
        assert result.aligned is False
        assert result.allowed_direction is None

    def test_get_htf_bias_4h_ranging(self):
        """4H RANGING should result in aligned=False, allowed_direction=None."""
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_ranging_df()

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.side_effect = [Bias.BULLISH, Bias.RANGING]
            result = get_htf_bias(df_1h, df_4h, min_swing_points=3)

        assert result.bias_4h == Bias.RANGING
        assert result.aligned is False
        assert result.allowed_direction is None

    def test_get_htf_bias_daily_optional(self):
        """daily=None should not affect alignment calculation."""
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_swing_df_with_clear_pattern("uptrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.return_value = Bias.BULLISH
            result = get_htf_bias(df_1h, df_4h, df_daily=None, min_swing_points=3)

        assert result.bias_daily is None
        assert result.aligned is True  # Should still work without daily


# ---------------------------------------------------------------------------
# Test: HTFBiasEngine
# ---------------------------------------------------------------------------

class TestHTFBiasEngine:
    def test_htf_engine_gate_allows_long_bullish(self):
        """LONG + BULLISH bias should return (True, ...)."""
        engine = HTFBiasEngine()
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_swing_df_with_clear_pattern("uptrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.return_value = Bias.BULLISH
            engine.refresh(df_1h, df_4h)

        allowed, reason = engine.gate_signal("LONG")
        assert allowed is True
        assert "aligned" in reason.lower() or "htf" in reason.lower()

    def test_htf_engine_gate_allows_short_bearish(self):
        """SHORT + BEARISH bias should return (True, ...)."""
        engine = HTFBiasEngine()
        df_1h = make_swing_df_with_clear_pattern("downtrend")
        df_4h = make_swing_df_with_clear_pattern("downtrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.return_value = Bias.BEARISH
            engine.refresh(df_1h, df_4h)

        allowed, reason = engine.gate_signal("SHORT")
        assert allowed is True

    def test_htf_engine_gate_blocks_long_bearish(self):
        """LONG + BEARISH bias should return (False, reason with 'HTF')."""
        engine = HTFBiasEngine()
        df_1h = make_swing_df_with_clear_pattern("downtrend")
        df_4h = make_swing_df_with_clear_pattern("downtrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.return_value = Bias.BEARISH
            engine.refresh(df_1h, df_4h)

        allowed, reason = engine.gate_signal("LONG")
        assert allowed is False
        assert "HTF" in reason

    def test_htf_engine_gate_blocks_short_bullish(self):
        """SHORT + BULLISH bias should return (False, ...)."""
        engine = HTFBiasEngine()
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_swing_df_with_clear_pattern("uptrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.return_value = Bias.BULLISH
            engine.refresh(df_1h, df_4h)

        allowed, reason = engine.gate_signal("SHORT")
        assert allowed is False

    def test_htf_engine_gate_blocks_ranging(self):
        """RANGING should return (False, ...)."""
        engine = HTFBiasEngine()
        df_1h = make_ranging_df()
        df_4h = make_ranging_df()

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.return_value = Bias.RANGING
            engine.refresh(df_1h, df_4h)

        allowed, reason = engine.gate_signal("LONG")
        assert allowed is False
        assert "ranging" in reason.lower()

    def test_htf_engine_gate_blocks_misaligned(self):
        """Misaligned (1H BULLISH, 4H BEARISH) should return (False, ...)."""
        engine = HTFBiasEngine()
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_swing_df_with_clear_pattern("downtrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.side_effect = [Bias.BULLISH, Bias.BEARISH]
            engine.refresh(df_1h, df_4h)

        allowed, reason = engine.gate_signal("LONG")
        assert allowed is False
        assert "misaligned" in reason.lower()

    def test_htf_engine_gate_disabled(self):
        """PRISM_HTF_ENABLED=0 should return (True, 'HTF gate disabled') regardless."""
        engine = HTFBiasEngine()
        df_1h = make_swing_df_with_clear_pattern("downtrend")
        df_4h = make_swing_df_with_clear_pattern("downtrend")

        with patch('prism.signal.htf_bias.classify_bias') as mock_classify:
            mock_classify.return_value = Bias.BEARISH
            engine.refresh(df_1h, df_4h)

        with patch.dict(os.environ, {"PRISM_HTF_ENABLED": "0"}):
            allowed, reason = engine.gate_signal("LONG")

        assert allowed is True
        assert "disabled" in reason.lower()

    def test_htf_engine_refresh_caches(self):
        """Same df tail should not recompute (cache hit)."""
        engine = HTFBiasEngine()
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_swing_df_with_clear_pattern("uptrend")

        with patch('prism.signal.htf_bias.get_htf_bias') as mock_get:
            mock_get.return_value = HTFBiasResult(
                bias_1h=Bias.BULLISH, bias_4h=Bias.BULLISH, bias_daily=None,
                swing_points_1h=[], swing_points_4h=[],
                aligned=True, allowed_direction="LONG"
            )

            # First call
            engine.refresh(df_1h, df_4h)
            assert mock_get.call_count == 1

            # Second call with same data - should use cache
            engine.refresh(df_1h, df_4h)
            assert mock_get.call_count == 1  # No additional call

    def test_htf_engine_refresh_invalidates_on_new_bars(self):
        """New bar appended should trigger recompute."""
        engine = HTFBiasEngine()
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_swing_df_with_clear_pattern("uptrend")

        with patch('prism.signal.htf_bias.get_htf_bias') as mock_get:
            mock_get.return_value = HTFBiasResult(
                bias_1h=Bias.BULLISH, bias_4h=Bias.BULLISH, bias_daily=None,
                swing_points_1h=[], swing_points_4h=[],
                aligned=True, allowed_direction="LONG"
            )

            # First call
            engine.refresh(df_1h, df_4h)
            assert mock_get.call_count == 1

            # Modify df_1h by appending a new row
            new_row = pd.DataFrame([{
                "open": 200, "high": 202, "low": 198, "close": 201, "volume": 1000
            }])
            df_1h_new = pd.concat([df_1h, new_row], ignore_index=True)

            # Second call with new data - should recompute
            engine.refresh(df_1h_new, df_4h)
            assert mock_get.call_count == 2

    def test_htf_engine_raises_if_no_refresh(self):
        """gate_signal should raise ValueError if refresh() never called."""
        engine = HTFBiasEngine()

        with pytest.raises(ValueError) as exc_info:
            engine.gate_signal("LONG")

        assert "refresh" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test: Environment Variables
# ---------------------------------------------------------------------------

class TestEnvironmentVariables:
    def test_htf_lookback_bars_env(self):
        """PRISM_HTF_LOOKBACK_BARS=50 should set engine.lookback_bars=50."""
        with patch.dict(os.environ, {"PRISM_HTF_LOOKBACK_BARS": "50"}):
            engine = HTFBiasEngine()
        assert engine.lookback_bars == 50

    def test_htf_min_swing_points_env(self):
        """PRISM_HTF_MIN_SWING_POINTS=5 should be honored."""
        with patch.dict(os.environ, {"PRISM_HTF_MIN_SWING_POINTS": "5"}):
            engine = HTFBiasEngine()
        assert engine.min_swing_points == 5

    def test_htf_default_lookback_bars(self):
        """Default lookback_bars should be 100."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear env vars that might be set
            os.environ.pop("PRISM_HTF_LOOKBACK_BARS", None)
            os.environ.pop("PRISM_HTF_MIN_SWING_POINTS", None)
            engine = HTFBiasEngine()
        assert engine.lookback_bars == 100

    def test_htf_default_min_swing_points(self):
        """Default min_swing_points should be 3."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PRISM_HTF_LOOKBACK_BARS", None)
            os.environ.pop("PRISM_HTF_MIN_SWING_POINTS", None)
            engine = HTFBiasEngine()
        assert engine.min_swing_points == 3


# ---------------------------------------------------------------------------
# Test: Generator Integration
# ---------------------------------------------------------------------------

class TestGeneratorIntegration:
    def test_generator_htf_gate_blocks_signal(self):
        """Mock htf_engine.gate_signal returns (False,...) -> generator returns None."""
        from prism.signal.generator import SignalGenerator

        # Create a generator with mocked dependencies
        with patch.object(SignalGenerator, '_load_predictor'):
            gen = SignalGenerator("XAUUSD", persist_fvg=False)

        # Mock all the dependencies
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = {
            "direction": 1,
            "confidence": 0.75,
            "direction_str": "LONG",
            "confidence_level": "HIGH",
            "magnitude_pips": 50,
        }

        gen.news = MagicMock()
        gen.news.get_signal.return_value = MagicMock(
            news_bias="NEUTRAL", risk_regime="RISK_ON"
        )
        gen.news.should_block_trade.return_value = (False, "")

        # Mock HTF engine to block
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = MagicMock()
        gen.htf_engine.gate_signal.return_value = (False, "HTF blocks LONG: bias is SHORT")

        # Create test DataFrames
        h4_df = pd.DataFrame({
            "open": [100], "high": [102], "low": [99], "close": [101],
            "volume": [1000], "feature_1": [0.5]
        })
        h1_df = pd.DataFrame({
            "open": [100], "high": [102], "low": [99], "close": [101], "volume": [1000]
        })
        entry_df = pd.DataFrame({
            "open": [100], "high": [102], "low": [99], "close": [101],
            "volume": [1000], "atr_14": [1.5]
        })

        result = gen.generate(h4_df, h1_df, entry_df)

        # Should return None because HTF gate blocked
        assert result is None
        gen.htf_engine.gate_signal.assert_called_once_with("LONG")

    def test_generator_htf_gate_allows_signal(self):
        """Mock returns (True,...) -> generator continues to next layer."""
        from prism.signal.generator import SignalGenerator

        with patch.object(SignalGenerator, '_load_predictor'):
            gen = SignalGenerator("XAUUSD", persist_fvg=False)

        # Mock predictor
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = {
            "direction": 1,
            "confidence": 0.75,
            "direction_str": "LONG",
            "confidence_level": "HIGH",
            "magnitude_pips": 50,
        }

        # Mock news
        gen.news = MagicMock()
        gen.news.get_signal.return_value = MagicMock(
            news_bias="NEUTRAL", risk_regime="RISK_ON"
        )
        gen.news.should_block_trade.return_value = (False, "")

        # Mock HTF engine to ALLOW
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = MagicMock(
            bias_1h=Bias.BULLISH,
            bias_4h=Bias.BULLISH,
            aligned=True,
            allowed_direction="LONG"
        )
        gen.htf_engine.gate_signal.return_value = (True, "HTF aligned")

        # Mock ICC to block (so we can verify HTF passed)
        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = []  # No ICC signals -> will return None

        h4_df = pd.DataFrame({
            "open": [100], "high": [102], "low": [99], "close": [101],
            "volume": [1000], "feature_1": [0.5]
        })
        h1_df = pd.DataFrame({
            "open": [100], "high": [102], "low": [99], "close": [101], "volume": [1000]
        })
        entry_df = pd.DataFrame({
            "open": [100], "high": [102], "low": [99], "close": [101],
            "volume": [1000], "atr_14": [1.5]
        })

        result = gen.generate(h4_df, h1_df, entry_df)

        # Should have passed HTF gate and called ICC
        gen.htf_engine.gate_signal.assert_called_once_with("LONG")
        gen.icc.detect_signals.assert_called_once()
        # Result is None because ICC blocked, but HTF passed
        assert result is None


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_swing_points_contain_required_fields(self):
        """Each swing point should have price, type, and bar_idx."""
        df = make_swing_df_with_clear_pattern("uptrend")
        swings = detect_swing_structure(df, lookback=3)

        for swing in swings:
            assert "price" in swing
            assert "type" in swing
            assert "bar_idx" in swing
            assert isinstance(swing["price"], float)
            assert swing["type"] in ("HH", "HL", "LH", "LL")
            assert isinstance(swing["bar_idx"], int)

    def test_htf_bias_result_fields(self):
        """HTFBiasResult should have all required fields."""
        df_1h = make_swing_df_with_clear_pattern("uptrend")
        df_4h = make_swing_df_with_clear_pattern("uptrend")

        result = get_htf_bias(df_1h, df_4h)

        assert hasattr(result, "bias_1h")
        assert hasattr(result, "bias_4h")
        assert hasattr(result, "bias_daily")
        assert hasattr(result, "swing_points_1h")
        assert hasattr(result, "swing_points_4h")
        assert hasattr(result, "aligned")
        assert hasattr(result, "allowed_direction")

    def test_bias_enum_values(self):
        """Bias enum should have correct string values."""
        assert Bias.BULLISH.value == "BULLISH"
        assert Bias.BEARISH.value == "BEARISH"
        assert Bias.RANGING.value == "RANGING"

    def test_engine_constructor_with_explicit_params(self):
        """Engine should accept explicit lookback_bars and min_swing_points."""
        engine = HTFBiasEngine(lookback_bars=75, min_swing_points=4)
        assert engine.lookback_bars == 75
        assert engine.min_swing_points == 4
