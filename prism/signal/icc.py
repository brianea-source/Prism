"""
prism/signal/icc.py
ICC Pattern Detector — Indication → Correction → Continuation
Methodology by @tradesbysci (Sci, 460K YouTube)
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_swing_points(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Mark swing highs and lows in OHLCV DataFrame.
    Returns df with added columns: is_swing_high, is_swing_low
    """
    df = df.copy()
    df["is_swing_high"] = False
    df["is_swing_low"] = False

    for i in range(lookback, len(df) - lookback):
        window_high = df["high"].iloc[i - lookback:i + lookback + 1]
        window_low = df["low"].iloc[i - lookback:i + lookback + 1]
        if df["high"].iloc[i] == window_high.max():
            df.at[df.index[i], "is_swing_high"] = True
        if df["low"].iloc[i] == window_low.min():
            df.at[df.index[i], "is_swing_low"] = True

    return df


def detect_icc_phase(
    df: pd.DataFrame,
    lookback: int = 20,
    correction_min: float = 0.30,
    correction_max: float = 0.65,
) -> str:
    """
    Detect current ICC phase on the last bar of df.

    Args:
        df: OHLCV DataFrame (at least lookback + 10 bars)
        lookback: bars to look back for swing detection
        correction_min: minimum retracement to qualify as correction (default 30%)
        correction_max: maximum retracement before trend considered broken (default 65%)

    Returns:
        'NONE' | 'INDICATION_BULL' | 'INDICATION_BEAR' |
        'CORRECTION_BULL' | 'CORRECTION_BEAR' |
        'CONTINUATION_LONG' | 'CONTINUATION_SHORT'
    """
    if len(df) < lookback + 10:
        return "NONE"

    recent = df.iloc[-(lookback + 10):]
    current_high = recent["high"].iloc[-1]
    current_low = recent["low"].iloc[-1]
    current_close = recent["close"].iloc[-1]

    period_high = recent["high"].iloc[:-1].max()
    period_low = recent["low"].iloc[:-1].min()

    # --- Indication detection ---
    new_hh = current_high > period_high  # New Higher High (bullish indication)
    new_ll = current_low < period_low    # New Lower Low (bearish indication)

    if not new_hh and not new_ll:
        return "NONE"

    if new_hh:
        # Measure from the swing low to new HH
        swing_low = recent["low"].min()
        indication_range = current_high - swing_low
        if indication_range <= 0:
            return "INDICATION_BULL"

        # Measure current pullback from the high
        pullback = current_high - current_close
        retracement_pct = pullback / indication_range

        if retracement_pct < correction_min:
            return "INDICATION_BULL"  # Still in indication, not corrected enough yet
        elif retracement_pct > correction_max:
            return "NONE"  # Trend broken, too deep a correction
        else:
            # In correction zone — check for continuation
            correction_low = recent["low"].iloc[-5:].min()
            if current_close > correction_low + (indication_range * 0.05):
                return "CONTINUATION_LONG"
            return "CORRECTION_BULL"

    if new_ll:
        # Measure from the swing high to new LL
        swing_high = recent["high"].max()
        indication_range = swing_high - current_low
        if indication_range <= 0:
            return "INDICATION_BEAR"

        pullback = current_close - current_low
        retracement_pct = pullback / indication_range

        if retracement_pct < correction_min:
            return "INDICATION_BEAR"
        elif retracement_pct > correction_max:
            return "NONE"
        else:
            correction_high = recent["high"].iloc[-5:].max()
            if current_close < correction_high - (indication_range * 0.05):
                return "CONTINUATION_SHORT"
            return "CORRECTION_BEAR"

    return "NONE"


def get_icc_entry(df: pd.DataFrame, instrument: str = "EURUSD") -> dict | None:
    """
    Returns an ICC entry signal dict if a valid Continuation is detected,
    otherwise None.

    Signal dict:
    {
        'phase': str,
        'direction': 'LONG' | 'SHORT',
        'entry': float,
        'sl': float,
        'correction_pct': float,
        'indication_range_pips': float,
        'aoi_nearby': bool,
    }
    """
    phase = detect_icc_phase(df)

    if phase not in ("CONTINUATION_LONG", "CONTINUATION_SHORT"):
        return None

    recent = df.iloc[-25:]
    current_close = df["close"].iloc[-1]
    current_high = df["high"].iloc[-1]
    current_low = df["low"].iloc[-1]

    pip_size = 0.01 if "XAU" in instrument or "JPY" in instrument else 0.0001

    if phase == "CONTINUATION_LONG":
        swing_low = recent["low"].min()
        indication_range = current_high - swing_low
        correction_low = recent["low"].iloc[-8:-1].min()
        sl = correction_low - (pip_size * 5)  # 5 pip buffer below correction low
        correction_pct = (current_high - correction_low) / indication_range if indication_range > 0 else 0

        return {
            "phase": phase,
            "direction": "LONG",
            "entry": current_close,
            "sl": round(sl, 5),
            "correction_pct": round(correction_pct, 3),
            "indication_range_pips": round(indication_range / pip_size, 1),
            "aoi_nearby": False,  # Will be set by AOIDetector
        }

    else:  # CONTINUATION_SHORT
        swing_high = recent["high"].max()
        indication_range = swing_high - current_low
        correction_high = recent["high"].iloc[-8:-1].max()
        sl = correction_high + (pip_size * 5)
        correction_pct = (correction_high - current_low) / indication_range if indication_range > 0 else 0

        return {
            "phase": phase,
            "direction": "SHORT",
            "entry": current_close,
            "sl": round(sl, 5),
            "correction_pct": round(correction_pct, 3),
            "indication_range_pips": round(indication_range / pip_size, 1),
            "aoi_nearby": False,
        }


class AOIDetector:
    """
    Areas of Interest detector — daily and weekly swing highs/lows.
    Sci's concept: trades at AOI confluence have highest probability.
    """

    def __init__(self, df_daily: pd.DataFrame):
        """
        df_daily: daily OHLCV DataFrame (at least 20 bars)
        """
        self.levels = self._build_levels(df_daily)

    def _build_levels(self, df: pd.DataFrame) -> list[dict]:
        levels = []
        if len(df) < 5:
            return levels

        # Daily swing highs and lows (last 20 days)
        for i in range(1, min(20, len(df) - 1)):
            row = df.iloc[-(i + 1)]
            prev = df.iloc[-(i + 2)] if i + 2 <= len(df) else None
            nxt = df.iloc[-i]

            if prev is not None:
                if row["high"] > prev["high"] and row["high"] > nxt["high"]:
                    levels.append({"price": row["high"], "type": "daily_high", "age_days": i})
                if row["low"] < prev["low"] and row["low"] < nxt["low"]:
                    levels.append({"price": row["low"], "type": "daily_low", "age_days": i})

        # Weekly high/low (last 5 weeks)
        if len(df) >= 35:
            for w in range(5):
                start = -(w + 1) * 5
                end = -w * 5 if w > 0 else None
                week = df.iloc[start:end]
                levels.append({"price": week["high"].max(), "type": "weekly_high", "age_days": (w + 1) * 5})
                levels.append({"price": week["low"].min(), "type": "weekly_low", "age_days": (w + 1) * 5})

        return levels

    def get_nearby_aoi(self, price: float, tolerance_pips: float = 20,
                       pip_size: float = 0.0001) -> list[dict]:
        """Returns list of AOI levels within tolerance of given price."""
        tolerance = tolerance_pips * pip_size
        return [lvl for lvl in self.levels if abs(lvl["price"] - price) <= tolerance]

    def is_at_aoi(self, price: float, tolerance_pips: float = 20,
                  pip_size: float = 0.0001) -> bool:
        return len(self.get_nearby_aoi(price, tolerance_pips, pip_size)) > 0
