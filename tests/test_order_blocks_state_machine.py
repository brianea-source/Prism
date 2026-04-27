"""Tests for Order Block state machine (Phase 6.A).

Tests cover: enum/dataclass, transition happy paths, terminal conditions, audit trail.
"""

import pandas as pd
import pytest

from prism.signal.order_blocks import OrderBlock, OrderBlockDetector, OrderBlockState


# --- Fixtures ---


def make_bar(open_: float, high: float, low: float, close: float) -> pd.Series:
    """Create a synthetic OHLC bar as pd.Series."""
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


def make_bullish_ob() -> OrderBlock:
    """Create a sample BULLISH Order Block at zone [1.1000, 1.1050]."""
    return OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BULLISH",
        high=1.1050,
        low=1.1000,
        midpoint=1.1025,
        formed_at="2024-01-15T08:00:00Z",
        formed_bar=100,
        displacement_size=15.0,
    )


def make_bearish_ob() -> OrderBlock:
    """Create a sample BEARISH Order Block at zone [1.1000, 1.1050]."""
    return OrderBlock(
        instrument="EURUSD",
        timeframe="H4",
        direction="BEARISH",
        high=1.1050,
        low=1.1000,
        midpoint=1.1025,
        formed_at="2024-01-15T08:00:00Z",
        formed_bar=100,
        displacement_size=15.0,
    )


# =============================================================================
# ENUM / DATACLASS TESTS (4 tests)
# =============================================================================


class TestEnumDataclass:
    """Tests for OrderBlockState enum and OrderBlock dataclass."""

    def test_ob_state_enum_values(self):
        """All 8 states have correct string values."""
        assert OrderBlockState.OB_FRESH.value == "OB_FRESH"
        assert OrderBlockState.OB_TESTED.value == "OB_TESTED"
        assert OrderBlockState.OB_RESPECTED.value == "OB_RESPECTED"
        assert OrderBlockState.OB_MITIGATED.value == "OB_MITIGATED"
        assert OrderBlockState.RB_FRESH.value == "RB_FRESH"
        assert OrderBlockState.RB_TESTED.value == "RB_TESTED"
        assert OrderBlockState.RB_RESPECTED.value == "RB_RESPECTED"
        assert OrderBlockState.CONSUMED.value == "CONSUMED"
        # Verify we have exactly 8 states
        assert len(OrderBlockState) == 8

    def test_ob_fresh_default_state(self):
        """New OrderBlock has state=OB_FRESH by default."""
        ob = make_bullish_ob()
        assert ob.state == OrderBlockState.OB_FRESH

    def test_is_rejection_block(self):
        """RB_FRESH/RB_TESTED/RB_RESPECTED → True; OB_* → False."""
        ob = make_bullish_ob()

        # OB states should not be rejection blocks
        ob.state = OrderBlockState.OB_FRESH
        assert ob.is_rejection_block is False
        ob.state = OrderBlockState.OB_TESTED
        assert ob.is_rejection_block is False
        ob.state = OrderBlockState.OB_RESPECTED
        assert ob.is_rejection_block is False
        ob.state = OrderBlockState.OB_MITIGATED
        assert ob.is_rejection_block is False
        ob.state = OrderBlockState.CONSUMED
        assert ob.is_rejection_block is False

        # RB states should be rejection blocks
        ob.state = OrderBlockState.RB_FRESH
        assert ob.is_rejection_block is True
        ob.state = OrderBlockState.RB_TESTED
        assert ob.is_rejection_block is True
        ob.state = OrderBlockState.RB_RESPECTED
        assert ob.is_rejection_block is True

    def test_effective_direction_flips(self):
        """BULLISH OB in RB_FRESH → effective_direction='BEARISH' and vice versa."""
        # BULLISH OB in OB state
        bullish_ob = make_bullish_ob()
        assert bullish_ob.effective_direction == "BULLISH"

        # BULLISH OB flipped to RB
        bullish_ob.state = OrderBlockState.RB_FRESH
        assert bullish_ob.effective_direction == "BEARISH"

        # BEARISH OB in OB state
        bearish_ob = make_bearish_ob()
        assert bearish_ob.effective_direction == "BEARISH"

        # BEARISH OB flipped to RB
        bearish_ob.state = OrderBlockState.RB_FRESH
        assert bearish_ob.effective_direction == "BULLISH"


# =============================================================================
# TRANSITION HAPPY PATH TESTS (7 tests)
# =============================================================================


class TestTransitionHappyPath:
    """Tests for state transition happy paths."""

    def test_transition_fresh_to_tested_bullish(self):
        """Bar touches BULLISH OB zone → OB_TESTED."""
        ob = make_bullish_ob()
        detector = OrderBlockDetector("EURUSD", "H4")

        # Bar that enters zone but closes above (touch, not break)
        # Zone is [1.1000, 1.1050], bar dips into zone but closes above
        bar = make_bar(open_=1.1060, high=1.1070, low=1.1020, close=1.1055)

        result = detector.transition(ob, bar, bar_idx=101)

        assert result is True
        assert ob.state == OrderBlockState.OB_TESTED
        assert ob.test_count == 1

    def test_transition_fresh_to_mitigated_bullish(self):
        """Bar close < block.low → OB_MITIGATED (break has priority over touch)."""
        ob = make_bullish_ob()
        detector = OrderBlockDetector("EURUSD", "H4")

        # Bar that closes below zone low (1.1000)
        bar = make_bar(open_=1.1020, high=1.1030, low=1.0980, close=1.0990)

        result = detector.transition(ob, bar, bar_idx=101)

        assert result is True
        assert ob.state == OrderBlockState.OB_MITIGATED
        assert ob.mitigated_at_bar == 101

    def test_transition_tested_to_respected(self):
        """Bar closes above zone after touch → OB_RESPECTED."""
        ob = make_bullish_ob()
        ob.state = OrderBlockState.OB_TESTED
        detector = OrderBlockDetector("EURUSD", "H4")

        # Bar closes above zone high (1.1050) - bounced up, respected
        bar = make_bar(open_=1.1040, high=1.1080, low=1.1030, close=1.1070)

        result = detector.transition(ob, bar, bar_idx=102)

        assert result is True
        assert ob.state == OrderBlockState.OB_RESPECTED

    def test_transition_tested_to_mitigated(self):
        """Bar close breaks zone while in TESTED → OB_MITIGATED."""
        ob = make_bullish_ob()
        ob.state = OrderBlockState.OB_TESTED
        detector = OrderBlockDetector("EURUSD", "H4")

        # Bar closes below zone low - mitigated while being tested
        bar = make_bar(open_=1.1010, high=1.1020, low=1.0970, close=1.0980)

        result = detector.transition(ob, bar, bar_idx=102)

        assert result is True
        assert ob.state == OrderBlockState.OB_MITIGATED
        assert ob.mitigated_at_bar == 102

    def test_transition_mitigated_to_rb_fresh(self):
        """Reversal bar within 5 bars → RB_FRESH."""
        ob = make_bullish_ob()
        ob.state = OrderBlockState.OB_MITIGATED
        ob.mitigated_at_bar = 101
        ob.age_bars = 102  # Will become 103 after transition
        detector = OrderBlockDetector("EURUSD", "H4")

        # Reversal: close re-enters zone from below (close > low)
        # For BULLISH OB, this means price came back up into zone
        bar = make_bar(open_=1.0990, high=1.1030, low=1.0980, close=1.1020)

        result = detector.transition(ob, bar, bar_idx=103)

        assert result is True
        assert ob.state == OrderBlockState.RB_FRESH
        assert ob.flipped_at_bar == 103

    def test_transition_rb_fresh_to_tested(self):
        """Bar enters zone from new side → RB_TESTED."""
        ob = make_bullish_ob()
        ob.state = OrderBlockState.RB_FRESH
        ob.flipped_at_bar = 103
        ob.age_bars = 105
        detector = OrderBlockDetector("EURUSD", "H4")

        # Bar touches zone (enters range)
        bar = make_bar(open_=1.1060, high=1.1070, low=1.1020, close=1.1055)

        result = detector.transition(ob, bar, bar_idx=106)

        assert result is True
        assert ob.state == OrderBlockState.RB_TESTED
        assert ob.test_count == 1

    def test_transition_rb_tested_to_respected(self):
        """Bar closes outside in flipped direction → RB_RESPECTED."""
        ob = make_bullish_ob()  # Original BULLISH, now RB (effective BEARISH)
        ob.state = OrderBlockState.RB_TESTED
        detector = OrderBlockDetector("EURUSD", "H4")

        # For original BULLISH OB now RB: respected if close < low (bounced down)
        bar = make_bar(open_=1.1010, high=1.1020, low=1.0970, close=1.0980)

        result = detector.transition(ob, bar, bar_idx=107)

        assert result is True
        assert ob.state == OrderBlockState.RB_RESPECTED


# =============================================================================
# TERMINAL CONDITIONS TESTS (4 tests)
# =============================================================================


class TestTerminalConditions:
    """Tests for terminal state transitions."""

    def test_transition_rb_fresh_consumed_after_timeout(self):
        """bar_idx - flipped_at_bar > 20 → CONSUMED.
        flipped_at_bar is an absolute bar index; timeout is checked as
        bar_idx - flipped_at_bar > 20 (both on same scale).
        """
        ob = make_bullish_ob()
        ob.state = OrderBlockState.RB_FRESH
        ob.flipped_at_bar = 1000  # Formed at absolute index 1000
        detector = OrderBlockDetector("EURUSD", "H4")

        # Bar that doesn't touch zone, 21 bars later (1021 - 1000 = 21 > 20)
        bar = make_bar(open_=1.1200, high=1.1220, low=1.1180, close=1.1210)

        result = detector.transition(ob, bar, bar_idx=1021)

        assert result is True
        assert ob.state == OrderBlockState.CONSUMED

    def test_transition_rb_tested_consumed_on_break(self):
        """Close breaks RB zone → CONSUMED."""
        ob = make_bullish_ob()  # Original BULLISH
        ob.state = OrderBlockState.RB_TESTED
        detector = OrderBlockDetector("EURUSD", "H4")

        # For original BULLISH OB in RB_TESTED: break if close > high (original direction wins)
        bar = make_bar(open_=1.1040, high=1.1080, low=1.1030, close=1.1070)

        result = detector.transition(ob, bar, bar_idx=108)

        assert result is True
        assert ob.state == OrderBlockState.CONSUMED

    def test_transition_respected_consumed_after_3_resets(self):
        """reset_cycles >= 3 on next RESPECTED → CONSUMED."""
        ob = make_bullish_ob()
        ob.state = OrderBlockState.OB_RESPECTED
        ob.reset_cycles = 3  # Already hit 3 resets
        detector = OrderBlockDetector("EURUSD", "H4")

        # Any bar - the transition from RESPECTED happens immediately
        bar = make_bar(open_=1.1100, high=1.1120, low=1.1090, close=1.1110)

        result = detector.transition(ob, bar, bar_idx=110)

        assert result is True
        assert ob.state == OrderBlockState.CONSUMED

    def test_transition_mitigated_consumed_no_reversal(self):
        """bar_idx - mitigated_at_bar > 5 with no reversal → CONSUMED.
        mitigated_at_bar is an absolute bar index; timeout is checked as
        bar_idx - mitigated_at_bar > 5 (both on same scale).
        """
        ob = make_bullish_ob()
        ob.state = OrderBlockState.OB_MITIGATED
        ob.mitigated_at_bar = 1000  # Mitigated at absolute index 1000
        detector = OrderBlockDetector("EURUSD", "H4")

        # Bar 6 steps later (1006 - 1000 = 6 > 5), no reversal — stays below zone
        bar = make_bar(open_=1.0950, high=1.0970, low=1.0940, close=1.0960)

        result = detector.transition(ob, bar, bar_idx=1006)

        assert result is True
        assert ob.state == OrderBlockState.CONSUMED


# =============================================================================
# AUDIT TRAIL TESTS (2 tests)
# =============================================================================


class TestAuditTrail:
    """Tests for transition logging and return values."""

    def test_transition_log_appended(self):
        """Each transition appends (bar_idx, old_state, new_state) tuple."""
        ob = make_bullish_ob()
        detector = OrderBlockDetector("EURUSD", "H4")

        # Transition 1: FRESH → TESTED
        bar1 = make_bar(open_=1.1060, high=1.1070, low=1.1020, close=1.1055)
        detector.transition(ob, bar1, bar_idx=101)

        assert len(ob.transition_log) == 1
        assert ob.transition_log[0] == (101, OrderBlockState.OB_FRESH, OrderBlockState.OB_TESTED)

        # Transition 2: TESTED → RESPECTED
        bar2 = make_bar(open_=1.1040, high=1.1080, low=1.1030, close=1.1070)
        detector.transition(ob, bar2, bar_idx=102)

        assert len(ob.transition_log) == 2
        assert ob.transition_log[1] == (102, OrderBlockState.OB_TESTED, OrderBlockState.OB_RESPECTED)

        # Transition 3: RESPECTED → FRESH (reset)
        bar3 = make_bar(open_=1.1100, high=1.1120, low=1.1090, close=1.1110)
        detector.transition(ob, bar3, bar_idx=103)

        assert len(ob.transition_log) == 3
        assert ob.transition_log[2] == (103, OrderBlockState.OB_RESPECTED, OrderBlockState.OB_FRESH)

    def test_transition_returns_true_on_change(self):
        """Returns True when state changed, False when no change."""
        ob = make_bullish_ob()
        detector = OrderBlockDetector("EURUSD", "H4")

        # Bar that doesn't touch zone - no transition
        bar_no_touch = make_bar(open_=1.1100, high=1.1120, low=1.1080, close=1.1110)
        result = detector.transition(ob, bar_no_touch, bar_idx=101)
        assert result is False
        assert ob.state == OrderBlockState.OB_FRESH

        # Bar that touches zone - transition happens
        bar_touch = make_bar(open_=1.1060, high=1.1070, low=1.1020, close=1.1055)
        result = detector.transition(ob, bar_touch, bar_idx=102)
        assert result is True
        assert ob.state == OrderBlockState.OB_TESTED


# =============================================================================
# BEARISH OB TEST (1 test)
# =============================================================================


class TestBearishOB:
    """Tests for BEARISH Order Block transitions."""

    def test_bearish_ob_touch_and_mitigation(self):
        """BEARISH OB: bar.close > block.high → OB_MITIGATED."""
        ob = make_bearish_ob()
        detector = OrderBlockDetector("EURUSD", "H4")

        # First, test touch: bar enters zone but closes inside
        bar_touch = make_bar(open_=1.0980, high=1.1030, low=1.0970, close=1.1020)
        result = detector.transition(ob, bar_touch, bar_idx=101)

        assert result is True
        assert ob.state == OrderBlockState.OB_TESTED

        # Now test mitigation: bar closes above zone high (1.1050)
        bar_mitigate = make_bar(open_=1.1040, high=1.1080, low=1.1030, close=1.1070)
        result = detector.transition(ob, bar_mitigate, bar_idx=102)

        assert result is True
        assert ob.state == OrderBlockState.OB_MITIGATED
        assert ob.mitigated_at_bar == 102


# =============================================================================
# ADDITIONAL EDGE CASE TESTS (2 tests to reach 20)
# =============================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_consumed_state_no_further_transitions(self):
        """CONSUMED state blocks all further transitions."""
        ob = make_bullish_ob()
        ob.state = OrderBlockState.CONSUMED
        detector = OrderBlockDetector("EURUSD", "H4")

        initial_age = ob.age_bars

        # Any bar - should not change state
        bar = make_bar(open_=1.1020, high=1.1030, low=1.0980, close=1.0990)
        result = detector.transition(ob, bar, bar_idx=200)

        assert result is False
        assert ob.state == OrderBlockState.CONSUMED
        assert ob.age_bars == initial_age + 1  # age still increments

    def test_rb_respected_resets_to_rb_fresh(self):
        """RB_RESPECTED → RB_FRESH when reset_cycles < 3."""
        ob = make_bullish_ob()
        ob.state = OrderBlockState.RB_RESPECTED
        ob.reset_cycles = 2
        detector = OrderBlockDetector("EURUSD", "H4")

        bar = make_bar(open_=1.1100, high=1.1120, low=1.1090, close=1.1110)
        result = detector.transition(ob, bar, bar_idx=110)

        assert result is True
        assert ob.state == OrderBlockState.RB_FRESH
        assert ob.reset_cycles == 3
