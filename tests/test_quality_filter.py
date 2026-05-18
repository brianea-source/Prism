"""Tests for ``prism.signal.quality_filter`` (Phase 8 Task 1.1, Stage 1).

Coverage areas:

* OTE zone math — happy path, both directions, boundary cases (0.62 / 0.79),
  degenerate inputs.
* FVG quality scoring — each sub-component isolated, then combined; kill-zone
  bonus capped at 1.0; defensive on malformed input.
* Candle confirmation — bullish/bearish engulfing, hammer / inverted hammer,
  EMA cross gating, no-signal cases, short dataframes.
* Liquidity pool selection — closest above for LONG, closest below for SHORT,
  None when no qualifying pool, accepts floats / dicts / objects.
* Integration with ``SignalGenerator`` — flag default-off yields identical
  signal count to baseline; flag-on can suppress signals when quality is low.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from prism.signal import quality_filter as qf
from prism.signal.quality_filter import (
    OTE_LOWER_FIB,
    OTE_UPPER_FIB,
    QualityCheckResult,
    apply_quality_filter,
    check_candle_confirmation,
    check_ote_zone,
    next_liquidity_pool,
    score_fvg_quality,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FVG:
    """Lightweight stand-in for FVGZone — keeps tests fast and dependency-free."""

    top: float
    bottom: float
    direction: str = "BULLISH"
    age_bars: int = 0
    retest_confirmed: bool = False

    @property
    def midline(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class _OB:
    """Stand-in for OrderBlock."""

    midpoint: float
    direction: str = "BULLISH"
    effective_direction: Optional[str] = None


def _ohlc_row(o: float, h: float, l: float, c: float) -> dict:
    return {"open": o, "high": h, "low": l, "close": c}


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# check_ote_zone
# ---------------------------------------------------------------------------


class TestOTEZone:
    def test_long_inside_zone(self):
        # leg 100 → 200, 65% retrace = 200 - 65 = 135
        in_zone, pct = check_ote_zone(135.0, swing_high=200.0, swing_low=100.0, direction="LONG")
        assert in_zone is True
        assert pct == pytest.approx(0.65, abs=1e-9)

    def test_long_outside_zone_shallow(self):
        # 30% retrace — too shallow for OTE
        in_zone, pct = check_ote_zone(170.0, 200.0, 100.0, "LONG")
        assert in_zone is False
        assert pct == pytest.approx(0.30, abs=1e-9)

    def test_long_outside_zone_deep(self):
        # 90% retrace — too deep
        in_zone, pct = check_ote_zone(110.0, 200.0, 100.0, "LONG")
        assert in_zone is False
        assert pct == pytest.approx(0.90, abs=1e-9)

    def test_long_lower_boundary_exact(self):
        # exactly 62% retrace -> in zone (closed interval)
        price = 200.0 - 0.62 * 100.0  # 138.0
        in_zone, pct = check_ote_zone(price, 200.0, 100.0, "LONG")
        assert in_zone is True
        assert pct == pytest.approx(OTE_LOWER_FIB, abs=1e-9)

    def test_long_upper_boundary_exact(self):
        price = 200.0 - 0.79 * 100.0  # 121.0
        in_zone, pct = check_ote_zone(price, 200.0, 100.0, "LONG")
        assert in_zone is True
        assert pct == pytest.approx(OTE_UPPER_FIB, abs=1e-9)

    def test_short_inside_zone(self):
        # leg 200 → 100, 70% retrace up = 100 + 70 = 170
        in_zone, pct = check_ote_zone(170.0, 200.0, 100.0, "SHORT")
        assert in_zone is True
        assert pct == pytest.approx(0.70, abs=1e-9)

    def test_short_outside_zone(self):
        in_zone, pct = check_ote_zone(120.0, 200.0, 100.0, "SHORT")
        assert in_zone is False
        assert pct == pytest.approx(0.20, abs=1e-9)

    def test_invalid_direction_returns_falsey(self):
        in_zone, pct = check_ote_zone(150.0, 200.0, 100.0, "SIDEWAYS")
        assert in_zone is False
        assert pct == 0.0

    def test_zero_leg(self):
        in_zone, pct = check_ote_zone(100.0, 100.0, 100.0, "LONG")
        assert in_zone is False
        assert pct == 0.0

    def test_inverted_leg(self):
        # swing_high < swing_low — degenerate, should not crash and not pass.
        in_zone, pct = check_ote_zone(150.0, 100.0, 200.0, "LONG")
        assert in_zone is False
        assert pct == 0.0


# ---------------------------------------------------------------------------
# score_fvg_quality
# ---------------------------------------------------------------------------


class TestFVGQuality:
    def test_size_dominates_when_only_signal(self):
        # Force age past horizon so freshness contributes 0, isolating size.
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=qf.FVG_AGE_HORIZON_BARS)
        score = score_fvg_quality(fvg, atr=5.0)
        # gap=10, atr=5 -> ratio 2.0 -> size sub capped at 1.0 -> 0.40 * 1.0
        assert score == pytest.approx(0.40, abs=1e-9)

    def test_zero_atr_yields_zero_size(self):
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=0)
        # ATR=0 -> size=0; freshness=1 -> 0.25; nothing else -> 0.25
        score = score_fvg_quality(fvg, atr=0.0)
        assert score == pytest.approx(0.25, abs=1e-9)

    def test_freshness_decays_to_zero(self):
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=qf.FVG_AGE_HORIZON_BARS)
        # Size sub maxed (atr small relative to gap), freshness=0.
        # 0.40 (size) + 0.0 (fresh) = 0.40
        score = score_fvg_quality(fvg, atr=5.0)
        assert score == pytest.approx(0.40, abs=1e-9)

    def test_freshness_full_at_age_zero(self):
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=0)
        # Pure freshness contribution alone (no atr -> size=0): 0.25
        # Plus size: gap=10/atr=10 -> ratio 1.0 -> size=1/1.5=0.667 -> 0.40 * 0.667 = 0.267
        score = score_fvg_quality(fvg, atr=10.0)
        # 0.267 + 0.25 = 0.517
        assert score == pytest.approx(0.40 * (1.0 / 1.5) + 0.25, abs=1e-3)

    def test_confluence_with_aligned_ob(self):
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=qf.FVG_AGE_HORIZON_BARS)  # freshness=0
        ob = _OB(midpoint=106.0, direction="BULLISH")  # midline=105, gap=10, dist=1
        # 0.40 (size) + 0.0 (fresh) + 0.25 (confluence) = 0.65
        score = score_fvg_quality(fvg, ob_list=[ob], atr=5.0)
        assert score == pytest.approx(0.65, abs=1e-9)

    def test_confluence_skips_misaligned_ob(self):
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=qf.FVG_AGE_HORIZON_BARS)
        ob = _OB(midpoint=106.0, direction="BEARISH")
        score = score_fvg_quality(fvg, ob_list=[ob], atr=5.0)
        # No confluence credit -> just size = 0.40
        assert score == pytest.approx(0.40, abs=1e-9)

    def test_confluence_respects_effective_direction(self):
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=qf.FVG_AGE_HORIZON_BARS)
        # Raw direction is bearish but effective_direction has flipped bullish
        # (typical after an OB break-of-structure flip).
        ob = _OB(midpoint=106.0, direction="BEARISH", effective_direction="BULLISH")
        score = score_fvg_quality(fvg, ob_list=[ob], atr=5.0)
        assert score == pytest.approx(0.65, abs=1e-9)

    def test_confluence_distance_too_far(self):
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=qf.FVG_AGE_HORIZON_BARS)
        # midline=105, gap=10, budget=30 -> ob at 200 is way outside
        ob = _OB(midpoint=200.0, direction="BULLISH")
        score = score_fvg_quality(fvg, ob_list=[ob], atr=5.0)
        assert score == pytest.approx(0.40, abs=1e-9)

    def test_retest_confirmed_adds_credit(self):
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=qf.FVG_AGE_HORIZON_BARS,
                   retest_confirmed=True)
        # size=0.40 + retest=0.10 = 0.50
        score = score_fvg_quality(fvg, atr=5.0)
        assert score == pytest.approx(0.50, abs=1e-9)

    def test_kill_zone_bonus_caps_at_one(self):
        # Construct a high-quality FVG: size maxed, fresh, OB confluent, retest.
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=0, retest_confirmed=True)
        ob = _OB(midpoint=105.0, direction="BULLISH")
        score_no_kz = score_fvg_quality(fvg, ob_list=[ob], atr=5.0, kill_zone=False)
        score_kz = score_fvg_quality(fvg, ob_list=[ob], atr=5.0, kill_zone=True)
        # No-KZ: 0.40 + 0.25 + 0.25 + 0.10 = 1.00 already
        assert score_no_kz == pytest.approx(1.0, abs=1e-9)
        # KZ bonus would push above 1.0 but is capped.
        assert score_kz == pytest.approx(1.0, abs=1e-9)

    def test_kill_zone_bonus_applies_below_cap(self):
        # Lower-quality FVG so the bonus actually shows up in the output.
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=qf.FVG_AGE_HORIZON_BARS)
        score_no_kz = score_fvg_quality(fvg, atr=5.0, kill_zone=False)  # 0.40
        score_kz = score_fvg_quality(fvg, atr=5.0, kill_zone=True)
        assert score_kz > score_no_kz
        assert score_kz == pytest.approx(0.40 * qf.FVG_KILL_ZONE_BONUS, abs=1e-9)

    def test_malformed_fvg_returns_zero(self):
        # Object with no top/bottom -> score 0.0, no exception.
        score = score_fvg_quality(object(), atr=5.0)
        assert score == 0.0

    def test_age_override(self):
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=0)
        # Override to past horizon
        score_old = score_fvg_quality(fvg, atr=5.0, age_bars=qf.FVG_AGE_HORIZON_BARS)
        score_fresh = score_fvg_quality(fvg, atr=5.0)  # uses fvg.age_bars=0
        assert score_fresh > score_old


# ---------------------------------------------------------------------------
# check_candle_confirmation
# ---------------------------------------------------------------------------


def _flat_then_engulf_long() -> pd.DataFrame:
    """20 flat bars below an EMA, then a bullish engulfing that closes above it."""
    base = [_ohlc_row(99.5, 100.0, 99.0, 99.6) for _ in range(20)]
    # Previous bearish candle
    prev = _ohlc_row(101.0, 101.2, 100.0, 100.2)
    # Bullish engulfing: opens below prev close, closes above prev open
    curr = _ohlc_row(100.0, 105.0, 99.8, 102.0)
    return _df(base + [prev, curr])


def _flat_then_engulf_short() -> pd.DataFrame:
    base = [_ohlc_row(100.5, 101.0, 100.0, 100.4) for _ in range(20)]
    prev = _ohlc_row(99.0, 100.0, 98.8, 99.8)  # bullish
    # Bearish engulfing: opens above prev close, closes below prev open
    curr = _ohlc_row(100.0, 100.2, 95.0, 97.0)
    return _df(base + [prev, curr])


def _hammer_long_df() -> pd.DataFrame:
    base = [_ohlc_row(99.5, 100.0, 99.0, 99.6) for _ in range(20)]
    prev = _ohlc_row(101.0, 101.5, 100.5, 101.0)  # any non-engulfing prev
    # Hammer: small body, long lower wick, minimal upper wick.
    # body = |101.5-101.0| = 0.5; lower_wick = 101.0-95.0 = 6.0 (>= 2x body);
    # upper_wick = 101.6-101.5 = 0.1 (<= body). Close 101.6 lifts above EMA.
    curr = _ohlc_row(101.0, 101.6, 95.0, 101.5)
    return _df(base + [prev, curr])


def _inverted_hammer_short_df() -> pd.DataFrame:
    base = [_ohlc_row(100.5, 101.0, 100.0, 100.4) for _ in range(20)]
    prev = _ohlc_row(99.0, 100.0, 98.0, 99.0)
    # Inverted hammer: small body, long upper wick, minimal lower wick.
    # body = |98.5-99.0| = 0.5; upper_wick = 105.0-99.0 = 6.0 (>= 2x body);
    # lower_wick = 98.5-98.4 = 0.1 (<= body). Close 98.5 drops below EMA.
    curr = _ohlc_row(99.0, 105.0, 98.4, 98.5)
    return _df(base + [prev, curr])


class TestCandleConfirmation:
    def test_bullish_engulfing_with_ema_pass(self):
        df = _flat_then_engulf_long()
        assert check_candle_confirmation(df, "LONG") is True

    def test_bearish_engulfing_with_ema_pass(self):
        df = _flat_then_engulf_short()
        assert check_candle_confirmation(df, "SHORT") is True

    def test_hammer_long(self):
        df = _hammer_long_df()
        assert check_candle_confirmation(df, "LONG") is True

    def test_inverted_hammer_short(self):
        df = _inverted_hammer_short_df()
        assert check_candle_confirmation(df, "SHORT") is True

    def test_no_reversal_pattern(self):
        # A perfectly flat dataframe — no engulfing, no hammer.
        df = _df([_ohlc_row(100.0, 100.5, 99.5, 100.0) for _ in range(25)])
        assert check_candle_confirmation(df, "LONG") is False
        assert check_candle_confirmation(df, "SHORT") is False

    def test_engulfing_but_wrong_ema_side(self):
        # Bullish engulfing but the close ends UP below the EMA -> fail.
        # Construct EMA-heavy candles above, then a small bullish engulfing
        # that doesn't cross.
        bars = [_ohlc_row(105.0, 106.0, 104.0, 105.5) for _ in range(25)]
        bars[-2] = _ohlc_row(105.5, 105.6, 104.0, 104.2)  # prev bearish, low close
        bars[-1] = _ohlc_row(104.0, 104.6, 103.8, 104.5)  # bullish engulfing but tiny
        df = _df(bars)
        # The current close (104.5) is below the EMA pinned at ~105 -> False.
        assert check_candle_confirmation(df, "LONG") is False

    def test_too_short_dataframe(self):
        df = _df([_ohlc_row(100.0, 101.0, 99.0, 100.5)])
        assert check_candle_confirmation(df, "LONG") is False

    def test_missing_columns(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert check_candle_confirmation(df, "LONG") is False

    def test_invalid_direction(self):
        df = _flat_then_engulf_long()
        assert check_candle_confirmation(df, "WAFFLE") is False

    def test_none_df(self):
        assert check_candle_confirmation(None, "LONG") is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# next_liquidity_pool
# ---------------------------------------------------------------------------


class TestNextLiquidityPool:
    def test_long_picks_closest_above(self):
        pools = [120.0, 150.0, 130.0]
        assert next_liquidity_pool(pools, "LONG", 100.0) == 120.0

    def test_short_picks_closest_below(self):
        pools = [80.0, 50.0, 70.0]
        assert next_liquidity_pool(pools, "SHORT", 100.0) == 80.0

    def test_no_qualifying_pool_long(self):
        pools = [80.0, 50.0]
        assert next_liquidity_pool(pools, "LONG", 100.0) is None

    def test_no_qualifying_pool_short(self):
        pools = [120.0, 150.0]
        assert next_liquidity_pool(pools, "SHORT", 100.0) is None

    def test_accepts_dict_pools(self):
        pools = [{"price": 120.0}, {"level": 130.0}, {"value": 115.0}]
        # 115 is the closest above
        assert next_liquidity_pool(pools, "LONG", 100.0) == 115.0

    def test_accepts_object_pools(self):
        class P:
            def __init__(self, p):
                self.price = p
        pools = [P(120.0), P(140.0)]
        assert next_liquidity_pool(pools, "LONG", 100.0) == 120.0

    def test_accepts_swing_dicts_with_high_low(self):
        pools = [{"high": 125.0}, {"low": 90.0}]
        # For LONG we only consider levels > 100 -> 125
        assert next_liquidity_pool(pools, "LONG", 100.0) == 125.0
        # For SHORT, only levels < 100 -> 90
        assert next_liquidity_pool(pools, "SHORT", 100.0) == 90.0

    def test_skips_unparseable_pools(self):
        pools = [object(), "not a number", None, 120.0]
        assert next_liquidity_pool(pools, "LONG", 100.0) == 120.0

    def test_invalid_direction_returns_none(self):
        assert next_liquidity_pool([120.0], "BUY", 100.0) is None

    def test_none_swings(self):
        assert next_liquidity_pool(None, "LONG", 100.0) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# apply_quality_filter
# ---------------------------------------------------------------------------


class TestApplyQualityFilter:
    def setup_method(self):
        # Always start each test with a clean env.
        for k in (
            "PRISM_QUALITY_FILTER_ENABLED",
            "PRISM_FVG_MIN_QUALITY",
            "PRISM_OTE_REQUIRED",
            "PRISM_REQUIRE_CANDLE_CONFIRM",
        ):
            os.environ.pop(k, None)

    def teardown_method(self):
        self.setup_method()

    def _good_inputs(self) -> dict:
        fvg = _FVG(top=110.0, bottom=100.0, age_bars=0, retest_confirmed=True)
        ob = _OB(midpoint=105.0, direction="BULLISH")
        return {
            "fvg_zone": fvg,
            "direction": "LONG",
            # Place entry inside OTE: leg 80→130, 70% retrace = 130 - 35 = 95
            "entry_price": 95.0,
            "swing_high": 130.0,
            "swing_low": 80.0,
            "entry_df": _flat_then_engulf_long(),
            "ob_list": [ob],
            "atr": 5.0,
            "kill_zone": True,
            "swing_points": [140.0, 145.0, 70.0],
        }

    def test_disabled_passthrough_populates_fields(self):
        # Flag off -> always passes, but fields are still populated for audit.
        res = apply_quality_filter(**self._good_inputs())
        assert isinstance(res, QualityCheckResult)
        assert res.enabled is False
        assert res.passed is True
        assert res.fvg_quality > 0.5
        assert res.in_ote is True
        assert res.candle_confirmed is True
        assert res.next_tp == 140.0  # closest above 95.0

    def test_enabled_all_pass(self):
        os.environ["PRISM_QUALITY_FILTER_ENABLED"] = "1"
        res = apply_quality_filter(**self._good_inputs())
        assert res.enabled is True
        assert res.passed is True
        assert res.reasons == []

    def test_enabled_blocks_on_low_quality(self):
        os.environ["PRISM_QUALITY_FILTER_ENABLED"] = "1"
        os.environ["PRISM_FVG_MIN_QUALITY"] = "0.95"  # impossible to clear
        # Disable candle requirement so we isolate the quality failure.
        os.environ["PRISM_REQUIRE_CANDLE_CONFIRM"] = "0"
        inputs = self._good_inputs()
        # Make the FVG mediocre.
        inputs["fvg_zone"] = _FVG(top=110.0, bottom=100.0, age_bars=20)
        inputs["ob_list"] = []
        res = apply_quality_filter(**inputs)
        assert res.passed is False
        assert any("fvg_quality" in r for r in res.reasons)

    def test_enabled_blocks_on_no_candle(self):
        os.environ["PRISM_QUALITY_FILTER_ENABLED"] = "1"
        inputs = self._good_inputs()
        # Flat OHLC -> no reversal pattern.
        inputs["entry_df"] = _df([_ohlc_row(100.0, 100.5, 99.5, 100.0) for _ in range(25)])
        res = apply_quality_filter(**inputs)
        assert res.passed is False
        assert any("candle" in r for r in res.reasons)

    def test_enabled_blocks_on_no_ote_when_required(self):
        os.environ["PRISM_QUALITY_FILTER_ENABLED"] = "1"
        os.environ["PRISM_OTE_REQUIRED"] = "1"
        os.environ["PRISM_REQUIRE_CANDLE_CONFIRM"] = "0"
        inputs = self._good_inputs()
        # 30% retrace -> outside OTE.
        inputs["entry_price"] = 130.0 - 0.30 * 50.0
        res = apply_quality_filter(**inputs)
        assert res.passed is False
        assert any("OTE" in r for r in res.reasons)

    def test_ote_not_required_by_default(self):
        os.environ["PRISM_QUALITY_FILTER_ENABLED"] = "1"
        os.environ["PRISM_REQUIRE_CANDLE_CONFIRM"] = "0"
        inputs = self._good_inputs()
        # 30% retrace -> outside OTE but should still pass (not required).
        inputs["entry_price"] = 130.0 - 0.30 * 50.0
        # Bump FVG quality threshold low so size/fresh dominate.
        res = apply_quality_filter(**inputs)
        assert res.passed is True
        assert res.in_ote is False  # informational only

    def test_env_overrides_min_quality(self):
        os.environ["PRISM_QUALITY_FILTER_ENABLED"] = "1"
        os.environ["PRISM_FVG_MIN_QUALITY"] = "0.0"
        os.environ["PRISM_REQUIRE_CANDLE_CONFIRM"] = "0"
        inputs = self._good_inputs()
        inputs["fvg_zone"] = _FVG(top=100.01, bottom=100.0, age_bars=qf.FVG_AGE_HORIZON_BARS)
        inputs["ob_list"] = []
        # Tiny / stale FVG would normally fail; threshold=0 lets it through.
        res = apply_quality_filter(**inputs)
        assert res.passed is True

    def test_swallows_subcheck_exceptions(self):
        os.environ["PRISM_QUALITY_FILTER_ENABLED"] = "1"
        os.environ["PRISM_REQUIRE_CANDLE_CONFIRM"] = "0"
        # entry_df missing columns -> candle helper returns False, doesn't raise.
        inputs = self._good_inputs()
        inputs["entry_df"] = pd.DataFrame({"open": [1.0], "close": [1.0]})
        res = apply_quality_filter(**inputs)
        # candle confirmation False, but not required -> still passes if quality OK.
        assert isinstance(res, QualityCheckResult)


# ---------------------------------------------------------------------------
# Integration: SignalGenerator behavior with the flag
# ---------------------------------------------------------------------------


class TestGeneratorIntegration:
    """Verify the generator's quality-filter wire-in respects the flag.

    We mock all the upstream layers so the generator deterministically reaches
    the quality gate. Then we toggle the env flag and assert:
        flag=off  -> baseline behavior (signal emitted)
        flag=on   -> signal suppressed when quality fails
    """

    def setup_method(self):
        os.environ.pop("PRISM_QUALITY_FILTER_ENABLED", None)

    def teardown_method(self):
        os.environ.pop("PRISM_QUALITY_FILTER_ENABLED", None)
        os.environ.pop("PRISM_FVG_MIN_QUALITY", None)

    def _build_generator(self):
        from prism.signal.generator import SignalGenerator
        gen = SignalGenerator("EURUSD", persist_fvg=False)

        # Mock predictor & detectors so generate() makes it to the quality gate.
        gen._predictor = MagicMock()
        gen._predictor.predict_latest.return_value = {
            "direction": 1,
            "direction_str": "LONG",
            "confidence": 0.80,
            "confidence_level": "HIGH",
            "magnitude_pips": 25.0,
        }
        # News: nothing blocking.
        gen.news = MagicMock()
        gen.news.get_signal.return_value = MagicMock(
            news_bias="NEUTRAL", risk_regime="RISK_ON"
        )
        gen.news.should_block_trade.return_value = (False, "")
        # HTF: allow.
        gen.htf_engine = MagicMock()
        gen.htf_engine.refresh.return_value = MagicMock(
            bias_1h=MagicMock(value="BULLISH"),
            bias_4h=MagicMock(value="BULLISH"),
            aligned=True,
            allowed_direction="LONG",
            swing_points_1h=[{"type": "HH"}, {"type": "HL"}, {"type": "HH"}],
            swing_points_4h=[{"type": "HH"}, {"type": "HL"}, {"type": "HH"}],
        )
        gen.htf_engine.gate_signal.return_value = (True, "")
        # ICC: continuation LONG.
        gen.icc = MagicMock()
        gen.icc.detect_signals.return_value = [
            {
                "phase": "CONTINUATION",
                "direction": "LONG",
                "correction_low": 1.0950,
                "correction_high": 1.1100,
                "leg_size": 0.0100,
            }
        ]
        # FVG: return a tradable BULLISH zone. Use the real FVGZone dataclass so
        # both attribute access and vars() work cleanly (MagicMock + __dict__
        # interplay is brittle on Python 3.14).
        from prism.signal.fvg import FVGZone
        fvg_zone = FVGZone(
            instrument="EURUSD", timeframe="H4", direction="BULLISH",
            top=1.1000, bottom=1.0990, midline=1.0995,
            formed_at="2026-01-01T00:00:00", formed_bar=0,
            age_bars=0, strength=2.0, retest_confirmed=True,
        )
        gen.fvg = MagicMock()
        gen.fvg.detect.return_value = []
        gen.fvg.check_entry_trigger.return_value = fvg_zone
        # OB detector stub
        gen.ob_detector = MagicMock()
        gen.ob_detector.get_nearest_ob.return_value = None
        return gen

    def _build_frames(self):
        # H4: 30 bars, one feature column so the model path doesn't trip.
        h4 = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=30, freq="4h"),
            "open": np.linspace(1.10, 1.10, 30),
            "high": np.linspace(1.105, 1.105, 30),
            "low": np.linspace(1.095, 1.095, 30),
            "close": np.linspace(1.10, 1.10, 30),
            "volume": np.ones(30),
            "feat1": np.zeros(30),
            "atr_14": np.full(30, 0.001),
        })
        h1 = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=60, freq="1h"),
            "open": np.full(60, 1.10),
            "high": np.full(60, 1.101),
            "low": np.full(60, 1.099),
            "close": np.full(60, 1.10),
            "volume": np.ones(60),
        })
        # M15 entry frame engineered for bullish engulfing + EMA cross at end.
        m15_rows = [
            _ohlc_row(1.0995, 1.1000, 1.0990, 1.0996) for _ in range(25)
        ]
        # Previous bearish then bullish engulfing closing above EMA
        m15_rows.append(_ohlc_row(1.0998, 1.0999, 1.0985, 1.0986))
        m15_rows.append(_ohlc_row(1.0985, 1.1030, 1.0984, 1.1020))
        m15 = pd.DataFrame(m15_rows)
        m15["datetime"] = pd.date_range("2026-01-01", periods=len(m15_rows), freq="15min")
        m15["volume"] = 1.0
        m15["atr_14"] = 0.002
        return h4, h1, m15

    def test_flag_off_signal_emitted(self):
        gen = self._build_generator()
        h4, h1, m15 = self._build_frames()
        signal = gen.generate(h4, h1, m15)
        assert signal is not None, "Baseline (flag off) should still emit signal"

    def test_flag_on_low_quality_suppresses(self):
        os.environ["PRISM_QUALITY_FILTER_ENABLED"] = "1"
        # Set min quality impossibly high so the gate must fail.
        os.environ["PRISM_FVG_MIN_QUALITY"] = "0.99"
        # Disable candle requirement so we isolate the quality block.
        os.environ["PRISM_REQUIRE_CANDLE_CONFIRM"] = "0"
        try:
            gen = self._build_generator()
            h4, h1, m15 = self._build_frames()
            signal = gen.generate(h4, h1, m15)
            assert signal is None, "Flag on + impossible quality should suppress"
        finally:
            os.environ.pop("PRISM_REQUIRE_CANDLE_CONFIRM", None)

    def test_flag_on_passes_when_quality_ok(self):
        os.environ["PRISM_QUALITY_FILTER_ENABLED"] = "1"
        os.environ["PRISM_FVG_MIN_QUALITY"] = "0.0"
        os.environ["PRISM_REQUIRE_CANDLE_CONFIRM"] = "0"
        try:
            gen = self._build_generator()
            h4, h1, m15 = self._build_frames()
            signal = gen.generate(h4, h1, m15)
            assert signal is not None
        finally:
            os.environ.pop("PRISM_REQUIRE_CANDLE_CONFIRM", None)
