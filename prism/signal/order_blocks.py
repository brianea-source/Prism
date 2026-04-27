"""Order Block state machine for PRISM Phase 6.A.

This module implements the OB/RB state machine with 8 states and 13 transition rules.
Phase 6.A scope: state machine only, no detect() or update_states() logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class OrderBlockState(str, Enum):
    """8-state lifecycle for Order Blocks and Rejection Blocks."""

    OB_FRESH = "OB_FRESH"
    OB_TESTED = "OB_TESTED"
    OB_RESPECTED = "OB_RESPECTED"
    OB_MITIGATED = "OB_MITIGATED"
    RB_FRESH = "RB_FRESH"
    RB_TESTED = "RB_TESTED"
    RB_RESPECTED = "RB_RESPECTED"
    CONSUMED = "CONSUMED"


@dataclass
class OrderBlock:
    """Represents an Order Block or Rejection Block zone."""

    instrument: str
    timeframe: str
    direction: str  # "BULLISH" or "BEARISH"
    high: float
    low: float
    midpoint: float
    formed_at: str  # ISO datetime
    formed_bar: int
    displacement_size: float  # pips

    state: OrderBlockState = OrderBlockState.OB_FRESH
    age_bars: int = 0
    mitigated_at_bar: Optional[int] = None
    flipped_at_bar: Optional[int] = None
    test_count: int = 0
    transition_log: list = field(default_factory=list)  # [(bar_idx, old_state, new_state), ...]
    reset_cycles: int = 0  # track how many RESPECTED→FRESH resets have happened

    @property
    def is_rejection_block(self) -> bool:
        """True if this block has flipped to a Rejection Block state."""
        return self.state.value.startswith("RB_")

    @property
    def is_active(self) -> bool:
        """True if this block is still tradeable (not consumed)."""
        return self.state != OrderBlockState.CONSUMED

    @property
    def is_terminal(self) -> bool:
        """True if this block has reached terminal CONSUMED state."""
        return self.state == OrderBlockState.CONSUMED

    @property
    def effective_direction(self) -> str:
        """Return flipped direction for RB, otherwise original direction."""
        if self.is_rejection_block:
            return "BEARISH" if self.direction == "BULLISH" else "BULLISH"
        return self.direction


class OrderBlockDetector:
    """Detects and manages Order Block lifecycle states."""

    def __init__(self, instrument: str, timeframe: str = "H4"):
        self.instrument = instrument
        self.timeframe = timeframe
        self.blocks: list[OrderBlock] = []

    def _touches_zone(self, block: OrderBlock, bar: pd.Series) -> bool:
        """Check if bar's range enters the block zone."""
        return bar["low"] <= block.high and bar["high"] >= block.low

    def _close_breaks_zone(self, block: OrderBlock, bar: pd.Series) -> bool:
        """Check if bar close breaks through the zone (mitigation)."""
        if block.direction == "BULLISH":
            # Bullish OB: mitigated when close < low (price broke down through zone)
            return bar["close"] < block.low
        else:
            # Bearish OB: mitigated when close > high (price broke up through zone)
            return bar["close"] > block.high

    def _close_outside_original_direction(self, block: OrderBlock, bar: pd.Series) -> bool:
        """Check if bar closed outside zone in the original direction (respected)."""
        if block.direction == "BULLISH":
            # Bullish OB: respected when close > high (bounced up)
            return bar["close"] > block.high
        else:
            # Bearish OB: respected when close < low (bounced down)
            return bar["close"] < block.low

    def _close_outside_flipped_direction(self, block: OrderBlock, bar: pd.Series) -> bool:
        """Check if bar closed outside zone in the flipped direction (RB respected)."""
        # For RB, effective direction is flipped, so "outside in flipped direction"
        # is outside in the OPPOSITE of original direction
        if block.direction == "BULLISH":
            # Original BULLISH OB became RB, effective direction is BEARISH
            # RB respected when close < low (bounced down in new bearish direction)
            return bar["close"] < block.low
        else:
            # Original BEARISH OB became RB, effective direction is BULLISH
            # RB respected when close > high (bounced up in new bullish direction)
            return bar["close"] > block.high

    def _rb_close_breaks_zone(self, block: OrderBlock, bar: pd.Series) -> bool:
        """Check if bar close breaks through the zone invalidating the RB.

        For RB, a "break" means price moves in the ORIGINAL direction (opposite of
        effective direction), proving the rejection was false.
        """
        if block.direction == "BULLISH":
            # Original BULLISH OB became RB (bearish)
            # RB broken if close > high (price went up, original bullish direction wins)
            return bar["close"] > block.high
        else:
            # Original BEARISH OB became RB (bullish)
            # RB broken if close < low (price went down, original bearish direction wins)
            return bar["close"] < block.low

    def _reversal_after_mitigation(self, block: OrderBlock, bar: pd.Series) -> bool:
        """Check if close re-enters zone from opposite side (reversal)."""
        if block.direction == "BULLISH":
            # Bullish OB was mitigated (close went below low)
            # Reversal: close now re-enters zone from below (close > low)
            return bar["close"] > block.low
        else:
            # Bearish OB was mitigated (close went above high)
            # Reversal: close now re-enters zone from above (close < high)
            return bar["close"] < block.high

    def transition(self, block: OrderBlock, bar: pd.Series, bar_idx: int) -> bool:
        """Apply state transition rules to a block based on current bar.

        Returns True if state changed, False otherwise.
        Implements all 13 transition rules with break-before-touch priority.
        """
        old_state = block.state
        new_state = old_state

        # Terminal state - no transitions possible
        if block.state == OrderBlockState.CONSUMED:
            block.age_bars += 1
            return False

        # Check transitions based on current state
        if block.state == OrderBlockState.OB_FRESH:
            # Rule 2: OB_FRESH → OB_MITIGATED (check break first - priority)
            if self._close_breaks_zone(block, bar):
                new_state = OrderBlockState.OB_MITIGATED
                block.mitigated_at_bar = bar_idx
            # Rule 1: OB_FRESH → OB_TESTED (touch without break)
            elif self._touches_zone(block, bar):
                new_state = OrderBlockState.OB_TESTED
                block.test_count += 1

        elif block.state == OrderBlockState.OB_TESTED:
            # Rule 4: OB_TESTED → OB_MITIGATED (check break first)
            if self._close_breaks_zone(block, bar):
                new_state = OrderBlockState.OB_MITIGATED
                block.mitigated_at_bar = bar_idx
            # Rule 3: OB_TESTED → OB_RESPECTED (close outside in original direction)
            elif self._close_outside_original_direction(block, bar):
                new_state = OrderBlockState.OB_RESPECTED

        elif block.state == OrderBlockState.OB_RESPECTED:
            # Rules 5/6: OB_RESPECTED → OB_FRESH (reset) or CONSUMED (max resets)
            if block.reset_cycles >= 3:
                # Rule 6: Too many resets, consume
                new_state = OrderBlockState.CONSUMED
            else:
                # Rule 5: Reset for next test
                new_state = OrderBlockState.OB_FRESH
                block.reset_cycles += 1

        elif block.state == OrderBlockState.OB_MITIGATED:
            # Rule 8: OB_MITIGATED → CONSUMED (timeout - no reversal within 5 bars)
            if block.mitigated_at_bar is not None and block.age_bars - block.mitigated_at_bar > 5:
                new_state = OrderBlockState.CONSUMED
            # Rule 7: OB_MITIGATED → RB_FRESH (reversal within 5 bars)
            elif block.mitigated_at_bar is not None and block.age_bars - block.mitigated_at_bar <= 5:
                if self._reversal_after_mitigation(block, bar):
                    new_state = OrderBlockState.RB_FRESH
                    block.flipped_at_bar = bar_idx

        elif block.state == OrderBlockState.RB_FRESH:
            # Rule 10: RB_FRESH → CONSUMED (timeout - no return within 20 bars)
            if block.flipped_at_bar is not None and block.age_bars - block.flipped_at_bar > 20:
                new_state = OrderBlockState.CONSUMED
            # Rule 9: RB_FRESH → RB_TESTED (bar returns to zone from new side)
            elif self._touches_zone(block, bar):
                new_state = OrderBlockState.RB_TESTED
                block.test_count += 1

        elif block.state == OrderBlockState.RB_TESTED:
            # Rule 12: RB_TESTED → CONSUMED (close breaks zone - RB invalid)
            # For RB, "break" is in the ORIGINAL direction (opposite of effective)
            # Original BULLISH OB → RB_TESTED: break if close > high (original bullish wins)
            # Original BEARISH OB → RB_TESTED: break if close < low (original bearish wins)
            if self._rb_close_breaks_zone(block, bar):
                new_state = OrderBlockState.CONSUMED
            # Rule 11: RB_TESTED → RB_RESPECTED (close outside in flipped direction)
            elif self._close_outside_flipped_direction(block, bar):
                new_state = OrderBlockState.RB_RESPECTED

        elif block.state == OrderBlockState.RB_RESPECTED:
            # Rule 13: RB_RESPECTED → RB_FRESH (reset) or CONSUMED (max resets)
            if block.reset_cycles >= 3:
                new_state = OrderBlockState.CONSUMED
            else:
                new_state = OrderBlockState.RB_FRESH
                block.reset_cycles += 1

        # Apply transition
        block.age_bars += 1

        if new_state != old_state:
            block.state = new_state
            block.transition_log.append((bar_idx, old_state, new_state))
            return True

        return False

    def detect(self, df: pd.DataFrame, min_displacement_pips: float = 10.0) -> list[OrderBlock]:
        """Detect new Order Blocks from OHLC data.

        Stub only - not implemented in Phase 6.A.
        """
        pass

    def update_states(self, df: pd.DataFrame) -> None:
        """Update states of all tracked blocks based on new bars.

        Stub only - not implemented in Phase 6.A.
        """
        pass

    def get_active_blocks(
        self, max_age_bars: int = 50, states: Optional[list[OrderBlockState]] = None
    ) -> list[OrderBlock]:
        """Return active blocks filtered by age and state.

        Stub only - not implemented in Phase 6.A.
        """
        pass

    def get_nearest_ob(self, price: float, direction: str) -> Optional[OrderBlock]:
        """Get the nearest Order Block to current price in given direction.

        Stub only - not implemented in Phase 6.A.
        """
        pass

    def distance_to_ob(self, price: float, direction: str) -> Optional[float]:
        """Calculate distance in pips to nearest OB in given direction.

        Stub only - not implemented in Phase 6.A.
        """
        pass

    def htf_priority_filter(self, candidates: list[OrderBlock]) -> list[OrderBlock]:
        """Filter candidates by HTF alignment priority.

        Stub only - not implemented in Phase 6.A.
        """
        pass
