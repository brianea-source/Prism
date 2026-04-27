"""Order Block detection and OB/RB lifecycle state machine (PRISM Phase 6).

Phase 6.A: 8-state enum, ``OrderBlock`` dataclass, ``transition()`` rules.
Phase 6.B: ``detect()``, ``update_states()``, distance / nearest / HTF helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from prism.execution.mt5_bridge import PIP_SIZE


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
    # First bar index where ``transition()`` applies (after displacement closes). If
    # None, ``formed_bar + 1`` is used so manual/test blocks behave as before.
    activation_bar: Optional[int] = None

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


_TF_RANK = {
    "MN1": 10,
    "W1": 9,
    "D1": 8,
    "DAILY": 8,
    "H4": 7,
    "H2": 6,
    "H1": 5,
    "M30": 4,
    "M15": 3,
    "M5": 2,
    "M1": 1,
}


def _timeframe_rank(tf: str) -> int:
    return _TF_RANK.get((tf or "").upper(), 0)


class OrderBlockDetector:
    """Detects and manages Order Block lifecycle states."""

    def __init__(self, instrument: str, timeframe: str = "H4"):
        self.instrument = instrument
        self.timeframe = timeframe
        self.blocks: list[OrderBlock] = []
        # Next bar index ``update_states`` will apply (0-based). Grows monotonically
        # for a lengthening ``df`` so replays stay idempotent.
        self._cursor: int = 0

    def _pip_size(self) -> float:
        return float(PIP_SIZE.get(self.instrument, 0.0001))

    @staticmethod
    def _require_ohlc(df: pd.DataFrame) -> None:
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

    def _bar_timestamp(self, df: pd.DataFrame, idx: int) -> str:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[idx]
            if hasattr(ts, "isoformat"):
                return ts.isoformat()
            return str(ts)
        return f"bar-{idx}"

    @staticmethod
    def _is_bearish(row: pd.Series) -> bool:
        return bool(row["close"] < row["open"])

    @staticmethod
    def _is_bullish(row: pd.Series) -> bool:
        return bool(row["close"] > row["open"])

    def _find_last_bearish(self, df: pd.DataFrame, before: int, lookback: int) -> Optional[int]:
        lo = max(0, before - lookback)
        for k in range(before - 1, lo - 1, -1):
            if self._is_bearish(df.iloc[k]):
                return k
        return None

    def _find_last_bullish(self, df: pd.DataFrame, before: int, lookback: int) -> Optional[int]:
        lo = max(0, before - lookback)
        for k in range(before - 1, lo - 1, -1):
            if self._is_bullish(df.iloc[k]):
                return k
        return None

    def _activation_idx(self, block: OrderBlock) -> int:
        if block.activation_bar is not None:
            return int(block.activation_bar)
        return int(block.formed_bar) + 1

    @staticmethod
    def _normalize_wanted_effective(direction: str) -> Optional[str]:
        m = {
            "LONG": "BULLISH",
            "BUY": "BULLISH",
            "BULLISH": "BULLISH",
            "SHORT": "BEARISH",
            "SELL": "BEARISH",
            "BEARISH": "BEARISH",
        }
        return m.get(direction.strip().upper())

    def _catch_up_new_blocks(self, df: pd.DataFrame, new_blocks: list[OrderBlock]) -> None:
        """Apply already-processed bars [activation, cursor) to blocks added mid-stream."""
        if self._cursor <= 0 or not new_blocks:
            return
        n = len(df)
        for ob in new_blocks:
            start = self._activation_idx(ob)
            for idx in range(start, min(self._cursor, n)):
                self.transition(ob, df.iloc[idx], idx)

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
            # Design note: reset fires unconditionally on the next bar entering this
            # state — no price check needed because OB_RESPECTED already required a
            # confirmed bounce. Each reset cycle allows the zone to absorb one more
            # test; after 3 cycles the zone is considered structurally consumed.
            if block.reset_cycles >= 3:
                # Rule 6: Too many resets, consume
                new_state = OrderBlockState.CONSUMED
            else:
                # Rule 5: Reset for next test
                new_state = OrderBlockState.OB_FRESH
                block.reset_cycles += 1

        elif block.state == OrderBlockState.OB_MITIGATED:
            # Rule 8: OB_MITIGATED → CONSUMED (timeout - no reversal within 5 bars)
            if block.mitigated_at_bar is not None and bar_idx - block.mitigated_at_bar > 5:
                new_state = OrderBlockState.CONSUMED
            # Rule 7: OB_MITIGATED → RB_FRESH (reversal within 5 bars)
            elif block.mitigated_at_bar is not None and bar_idx - block.mitigated_at_bar <= 5:
                if self._reversal_after_mitigation(block, bar):
                    new_state = OrderBlockState.RB_FRESH
                    block.flipped_at_bar = bar_idx

        elif block.state == OrderBlockState.RB_FRESH:
            # Rule 10: RB_FRESH → CONSUMED (timeout - no return within 20 bars)
            if block.flipped_at_bar is not None and bar_idx - block.flipped_at_bar > 20:
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
        """Scan for Order Blocks and append new ones in ``OB_FRESH`` state.

        - **Bullish OB:** last bearish candle before bullish displacement
          (impulse high clears the zone high by at least ``min_displacement_pips``).
        - **Bearish OB:** last bullish candle before bearish displacement
          (impulse low clears the zone low by at least ``min_displacement_pips``).

        Returns the list of blocks newly appended in this call (same objects as in
        ``self.blocks``). Idempotent per (direction, ``formed_bar``): at most one OB
        per opposing-candle index per scan.
        """
        if df.empty:
            return []
        self._require_ohlc(df)
        if len(df) < 3:
            return []

        pip = self._pip_size()
        min_move = float(min_displacement_pips) * pip
        lookback = 50
        new_blocks: list[OrderBlock] = []
        # Seed with already-tracked blocks so repeated detect() calls don't dup.
        seen: set[tuple[str, int]] = {(b.direction, b.formed_bar) for b in self.blocks}

        for disp_end in range(2, len(df)):
            # --- Bullish OB (demand): last bearish before upward displacement ---
            k_bear = self._find_last_bearish(df, disp_end, lookback)
            if k_bear is not None:
                key = ("BULLISH", k_bear)
                if key not in seen:
                    row_k = df.iloc[k_bear]
                    hi = float(df.iloc[k_bear + 1 : disp_end + 1]["high"].max())
                    lo_k = float(row_k["low"])
                    move = hi - lo_k
                    if move >= min_move and hi > float(row_k["high"]):
                        seen.add(key)
                        high = float(row_k["high"])
                        low = float(row_k["low"])
                        ob = OrderBlock(
                            instrument=self.instrument,
                            timeframe=self.timeframe,
                            direction="BULLISH",
                            high=high,
                            low=low,
                            midpoint=(high + low) / 2.0,
                            formed_at=self._bar_timestamp(df, k_bear),
                            formed_bar=int(k_bear),
                            displacement_size=round(move / pip, 4),
                            state=OrderBlockState.OB_FRESH,
                            activation_bar=int(disp_end) + 1,
                        )
                        self.blocks.append(ob)
                        new_blocks.append(ob)

            # --- Bearish OB (supply): last bullish before downward displacement ---
            k_bull = self._find_last_bullish(df, disp_end, lookback)
            if k_bull is not None:
                key = ("BEARISH", k_bull)
                if key not in seen:
                    row_k = df.iloc[k_bull]
                    lo = float(df.iloc[k_bull + 1 : disp_end + 1]["low"].min())
                    hi_k = float(row_k["high"])
                    move = hi_k - lo
                    if move >= min_move and lo < float(row_k["low"]):
                        seen.add(key)
                        high = float(row_k["high"])
                        low = float(row_k["low"])
                        ob = OrderBlock(
                            instrument=self.instrument,
                            timeframe=self.timeframe,
                            direction="BEARISH",
                            high=high,
                            low=low,
                            midpoint=(high + low) / 2.0,
                            formed_at=self._bar_timestamp(df, k_bull),
                            formed_bar=int(k_bull),
                            displacement_size=round(move / pip, 4),
                            state=OrderBlockState.OB_FRESH,
                            activation_bar=int(disp_end) + 1,
                        )
                        self.blocks.append(ob)
                        new_blocks.append(ob)

        self._catch_up_new_blocks(df, new_blocks)
        return new_blocks

    def update_states(self, df: pd.DataFrame) -> None:
        """Walk ``df`` from ``_cursor`` forward, calling ``transition()`` once per bar per block.

        Call after ``detect()`` (or on its own when blocks already exist). Safe to call
        repeatedly with the same ``df`` length: already-applied bars are skipped.
        """
        if df.empty:
            return
        self._require_ohlc(df)
        n = len(df)
        for idx in range(self._cursor, n):
            bar = df.iloc[idx]
            for block in self.blocks:
                if idx >= self._activation_idx(block):
                    self.transition(block, bar, idx)
        self._cursor = n

    def get_active_blocks(
        self, max_age_bars: int = 50, states: Optional[list[OrderBlockState]] = None
    ) -> list[OrderBlock]:
        """Return non-``CONSUMED`` blocks with ``age_bars <= max_age_bars``, optionally by state."""
        out: list[OrderBlock] = []
        for b in self.blocks:
            if b.state == OrderBlockState.CONSUMED:
                continue
            if states is not None and b.state not in states:
                continue
            if b.age_bars > max_age_bars:
                continue
            out.append(b)
        return out

    def get_nearest_ob(self, price: float, direction: str) -> Optional[OrderBlock]:
        """Nearest active block whose ``effective_direction`` matches trade *direction*."""
        want = self._normalize_wanted_effective(direction)
        if want is None:
            return None
        candidates = [b for b in self.get_active_blocks() if b.effective_direction == want]
        if not candidates:
            return None
        pip = self._pip_size()

        def dist_mid(b: OrderBlock) -> float:
            return abs(float(price) - b.midpoint) / pip

        return min(candidates, key=dist_mid)

    def distance_to_ob(self, price: float, direction: str) -> Optional[float]:
        """Distance in pips from *price* to the midpoint of the nearest matching block."""
        ob = self.get_nearest_ob(price, direction)
        if ob is None:
            return None
        pip = self._pip_size()
        return round(abs(float(price) - ob.midpoint) / pip, 4)

    def htf_priority_filter(self, candidates: list[OrderBlock]) -> list[OrderBlock]:
        """When midpoints sit within 5 pips, keep the highest-``timeframe`` block only."""
        if not candidates:
            return []
        pip = self._pip_size()
        tol = 5.0 * pip
        ordered = sorted(candidates, key=lambda b: (-_timeframe_rank(b.timeframe), b.formed_bar))
        kept: list[OrderBlock] = []
        for c in ordered:
            if any(abs(c.midpoint - k.midpoint) <= tol for k in kept):
                continue
            kept.append(c)
        return kept
