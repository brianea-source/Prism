"""
HTF Bias Engine — Higher Timeframe Direction Lock.
Implements finastictrading's "1H is your compass" rule extended to 4H.

Locks trade direction from 1H and 4H swing structure BEFORE any 5M signal
generation. A signal is only valid if it aligns with both 1H and 4H bias.
"""
import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Bias(str, Enum):
    """Market bias classification based on swing structure."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"


@dataclass
class HTFBiasResult:
    """Result of HTF bias analysis across timeframes."""
    bias_1h: Bias
    bias_4h: Bias
    bias_daily: Optional[Bias]  # Optional — not required for gating
    swing_points_1h: list       # [{"price": float, "type": "HH"|"HL"|"LH"|"LL", "bar_idx": int}, ...]
    swing_points_4h: list
    aligned: bool               # True if 1H and 4H agree (both BULLISH or both BEARISH)
    allowed_direction: Optional[str]  # "LONG" | "SHORT" | None (if ranging or misaligned)


def detect_swing_structure(df: pd.DataFrame, lookback: int = 3) -> list:
    """
    Detect the last N swing highs and lows.
    Returns list of swing points with type: HH, HL, LH, LL.

    A swing high at index i: high[i] > max(high[i-lookback:i]) AND high[i] > max(high[i+1:i+lookback+1])
    A swing low at index i: low[i] < min(low[i-lookback:i]) AND low[i] < min(low[i+1:i+lookback+1])

    Swing classification:
    - HH = new swing high > previous swing high
    - HL = new swing low > previous swing low
    - LH = new swing high < previous swing high
    - LL = new swing low < previous swing low
    """
    if len(df) < 2 * lookback + 1:
        return []

    # Ensure columns are lowercase
    high = df["high"].values
    low = df["low"].values

    # Check for NaN values
    if np.isnan(high).any() or np.isnan(low).any():
        logger.warning("NaN values in HTF data — swing detection returning empty")
        return []

    swing_highs = []  # (bar_idx, price)
    swing_lows = []   # (bar_idx, price)

    # Detect swing highs and lows
    for i in range(lookback, len(df) - lookback):
        # Check for swing high
        left_max = np.max(high[i - lookback:i])
        right_max = np.max(high[i + 1:i + lookback + 1])
        if high[i] > left_max and high[i] > right_max:
            swing_highs.append((i, float(high[i])))

        # Check for swing low
        left_min = np.min(low[i - lookback:i])
        right_min = np.min(low[i + 1:i + lookback + 1])
        if low[i] < left_min and low[i] < right_min:
            swing_lows.append((i, float(low[i])))

    # Classify swing points
    swing_points = []

    # Classify swing highs (HH or LH)
    for idx, (bar_idx, price) in enumerate(swing_highs):
        if idx == 0:
            # First swing high - no comparison, skip classification
            continue
        prev_price = swing_highs[idx - 1][1]
        swing_type = "HH" if price > prev_price else "LH"
        swing_points.append({
            "price": price,
            "type": swing_type,
            "bar_idx": bar_idx,
        })

    # Classify swing lows (HL or LL)
    for idx, (bar_idx, price) in enumerate(swing_lows):
        if idx == 0:
            # First swing low - no comparison, skip classification
            continue
        prev_price = swing_lows[idx - 1][1]
        swing_type = "HL" if price > prev_price else "LL"
        swing_points.append({
            "price": price,
            "type": swing_type,
            "bar_idx": bar_idx,
        })

    # Sort by bar_idx for chronological order
    swing_points.sort(key=lambda x: x["bar_idx"])

    return swing_points


def classify_bias(swing_points: list, min_swing_points: int = 3) -> Bias:
    """
    Classify trend bias from swing points.
    - HH + HL = BULLISH
    - LH + LL = BEARISH
    - Mixed or insufficient = RANGING

    Need at least min_swing_points to classify (checking last 2 highs and lows).
    """
    if len(swing_points) < min_swing_points:
        return Bias.RANGING

    # Get the most recent swing points
    recent = swing_points[-min_swing_points:]

    # Count swing types
    types = [p["type"] for p in recent]

    hh_count = types.count("HH")
    hl_count = types.count("HL")
    lh_count = types.count("LH")
    ll_count = types.count("LL")

    # BULLISH: Need both HH and HL pattern
    if hh_count > 0 and hl_count > 0 and lh_count == 0 and ll_count == 0:
        return Bias.BULLISH

    # BEARISH: Need both LH and LL pattern
    if lh_count > 0 and ll_count > 0 and hh_count == 0 and hl_count == 0:
        return Bias.BEARISH

    # Mixed or insufficient
    return Bias.RANGING


def get_htf_bias(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_daily: Optional[pd.DataFrame] = None,
    min_swing_points: int = 3,
) -> HTFBiasResult:
    """
    Main entry point: compute HTF bias from 1H and 4H bars.
    """
    # Detect swing structure
    swing_points_1h = detect_swing_structure(df_1h)
    swing_points_4h = detect_swing_structure(df_4h)

    # Classify bias for each timeframe
    bias_1h = classify_bias(swing_points_1h, min_swing_points)
    bias_4h = classify_bias(swing_points_4h, min_swing_points)

    # Optional daily bias
    bias_daily = None
    if df_daily is not None and len(df_daily) > 0:
        swing_points_daily = detect_swing_structure(df_daily)
        bias_daily = classify_bias(swing_points_daily, min_swing_points)

    # Alignment check: both 1H and 4H must be BULLISH or both BEARISH
    aligned = (
        bias_1h == bias_4h and
        bias_1h in (Bias.BULLISH, Bias.BEARISH)
    )

    # Allowed direction
    allowed_direction: Optional[str] = None
    if aligned:
        if bias_1h == Bias.BULLISH:
            allowed_direction = "LONG"
        elif bias_1h == Bias.BEARISH:
            allowed_direction = "SHORT"

    return HTFBiasResult(
        bias_1h=bias_1h,
        bias_4h=bias_4h,
        bias_daily=bias_daily,
        swing_points_1h=swing_points_1h,
        swing_points_4h=swing_points_4h,
        aligned=aligned,
        allowed_direction=allowed_direction,
    )


class HTFBiasEngine:
    """
    Stateful wrapper for HTF bias — caches results per session.
    """

    def __init__(self, lookback_bars: Optional[int] = None, min_swing_points: Optional[int] = None):
        # Read from environment with defaults
        default_lookback = int(os.getenv("PRISM_HTF_LOOKBACK_BARS", "100"))
        default_min_swing = int(os.getenv("PRISM_HTF_MIN_SWING_POINTS", "3"))

        self.lookback_bars = lookback_bars if lookback_bars is not None else default_lookback
        self.min_swing_points = min_swing_points if min_swing_points is not None else default_min_swing
        self._cache_key: Optional[str] = None
        self._cached_result: Optional[HTFBiasResult] = None

    def _compute_cache_key(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> str:
        """Compute cache key from df tails."""
        # Use last 5 close values from each df
        tail_1h = df_1h.tail(5)["close"].values.tolist() if len(df_1h) >= 5 else df_1h["close"].values.tolist()
        tail_4h = df_4h.tail(5)["close"].values.tolist() if len(df_4h) >= 5 else df_4h["close"].values.tolist()
        combined = tuple(tail_1h + tail_4h)
        return str(hash(combined))

    def refresh(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> HTFBiasResult:
        """Recompute bias (call once per kill zone start or hourly)."""
        # Check cache
        cache_key = self._compute_cache_key(df_1h, df_4h)

        if cache_key == self._cache_key and self._cached_result is not None:
            logger.debug("HTF bias cache hit")
            return self._cached_result

        # Limit to lookback_bars
        df_1h_limited = df_1h.tail(self.lookback_bars) if len(df_1h) > self.lookback_bars else df_1h
        df_4h_limited = df_4h.tail(self.lookback_bars) if len(df_4h) > self.lookback_bars else df_4h

        # Compute fresh result
        result = get_htf_bias(df_1h_limited, df_4h_limited, min_swing_points=self.min_swing_points)

        # Update cache
        self._cache_key = cache_key
        self._cached_result = result

        logger.debug(f"HTF bias computed: 1H={result.bias_1h.value}, 4H={result.bias_4h.value}, aligned={result.aligned}")

        return result

    def gate_signal(self, direction: str) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        - allowed=True if direction matches HTF bias
        - allowed=False with reason if misaligned or ranging

        Raises ValueError if refresh() has never been called.
        """
        # Check if HTF gating is disabled
        if os.getenv("PRISM_HTF_ENABLED", "1") == "0":
            return (True, "HTF gate disabled")

        # Check if we have a cached result
        if self._cached_result is None:
            raise ValueError("HTFBiasEngine.refresh() must be called before gate_signal()")

        result = self._cached_result

        # Check alignment
        if not result.aligned:
            if result.bias_1h == Bias.RANGING or result.bias_4h == Bias.RANGING:
                return (False, f"HTF ranging: 1H={result.bias_1h.value}, 4H={result.bias_4h.value}")
            return (False, f"HTF misaligned: 1H={result.bias_1h.value}, 4H={result.bias_4h.value}")

        # Check direction match
        if result.allowed_direction == direction:
            return (True, "HTF aligned")

        # Direction mismatch
        return (False, f"HTF blocks {direction}: bias is {result.allowed_direction}")
