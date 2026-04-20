"""
prism/signal/icc.py
-------------------
ICC (Indication → Correction → Continuation) pattern detector.

Framework derived from @tradesbysci methodology.

Phase sequence:
    NONE
    → INDICATION_BULL   (new higher-high vs lookback window)
    → INDICATION_BEAR   (new lower-low vs lookback window)
    → CORRECTION_BULL   (30–65% retracement of bullish indication range)
    → CORRECTION_BEAR   (30–65% retracement of bearish indication range)
    → CONTINUATION_LONG  (price closes back above correction low)
    → CONTINUATION_SHORT (price closes back below correction high)

Entry signals include SL placement at correction extreme and AOI confluence flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORRECTION_MIN_PCT = 0.30   # minimum retracement to qualify as correction
CORRECTION_MAX_PCT = 0.65   # maximum retracement (beyond = indication failure)
AOI_TOLERANCE_PIPS_DEFAULT = 20.0


# ---------------------------------------------------------------------------
# Swing point detection
# ---------------------------------------------------------------------------

def detect_swing_points(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Identify swing highs and swing lows in a price DataFrame.

    A swing high is a bar whose high is strictly greater than all highs
    in the ``lookback`` bars before AND after it.  Swing lows are symmetric.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: high, low.
    lookback : int
        Number of bars on each side of the candidate bar.

    Returns
    -------
    pd.DataFrame
        Input df with two new boolean columns:
        ``swing_high`` and ``swing_low``.
    """
    df = df.copy()
    highs = df["high"].values
    lows  = df["low"].values
    n = len(df)

    swing_high = np.zeros(n, dtype=bool)
    swing_low  = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        left_h  = highs[i - lookback: i]
        right_h = highs[i + 1: i + lookback + 1]
        if highs[i] > left_h.max() and highs[i] > right_h.max():
            swing_high[i] = True

        left_l  = lows[i - lookback: i]
        right_l = lows[i + 1: i + lookback + 1]
        if lows[i] < left_l.min() and lows[i] < right_l.min():
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"]  = swing_low
    return df


# ---------------------------------------------------------------------------
# ICC phase detection
# ---------------------------------------------------------------------------

def detect_icc_phase(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Detect the current ICC phase for the most recent bar in ``df``.

    Evaluates the last ``lookback`` bars to determine which phase the
    market is in.  Returns the phase string for the *current* bar.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (must have: open, high, low, close columns).
        Should already have ``swing_high`` / ``swing_low`` columns from
        ``detect_swing_points``; if absent they will be computed.
    lookback : int
        Bars to look back when classifying the indication range.

    Returns
    -------
    str
        One of: 'NONE', 'INDICATION_BULL', 'INDICATION_BEAR',
                'CORRECTION_BULL', 'CORRECTION_BEAR',
                'CONTINUATION_LONG', 'CONTINUATION_SHORT'
    """
    if len(df) < lookback + 2:
        return "NONE"

    if "swing_high" not in df.columns or "swing_low" not in df.columns:
        df = detect_swing_points(df, lookback=max(3, lookback // 4))

    window = df.iloc[-(lookback + 1):]
    cur    = window.iloc[-1]
    hist   = window.iloc[:-1]

    period_high = hist["high"].max()
    period_low  = hist["low"].min()

    close = cur["close"]

    # Indication: new structural extreme
    if close > period_high:
        indication = "BULL"
        indication_range = period_high - period_low
        indication_start = period_low
    elif close < period_low:
        indication = "BEAR"
        indication_range = period_high - period_low
        indication_start = period_high
    else:
        # Within the prior range — check for correction or continuation
        # Find the most recent swing high/low to determine indication direction
        last_sh = hist[hist["swing_high"]].iloc[-1]["high"] if hist["swing_high"].any() else None
        last_sl = hist[hist["swing_low"]].iloc[-1]["low"]   if hist["swing_low"].any() else None

        if last_sh is None and last_sl is None:
            return "NONE"

        # Determine dominant direction by most recent structural extreme
        if last_sh is not None and last_sl is not None:
            idx_sh = hist[hist["swing_high"]].index[-1]
            idx_sl = hist[hist["swing_low"]].index[-1]
            dominant = "BULL" if idx_sh > idx_sl else "BEAR"
        elif last_sh is not None:
            dominant = "BULL"
        else:
            dominant = "BEAR"

        if dominant == "BULL":
            indication_range = (last_sh or period_high) - period_low
            retracement = ((last_sh or period_high) - close) / (indication_range + 1e-12)
            if CORRECTION_MIN_PCT <= retracement <= CORRECTION_MAX_PCT:
                return "CORRECTION_BULL"
            if retracement < CORRECTION_MIN_PCT:
                return "INDICATION_BULL"  # still in indication leg
        else:
            indication_range = period_high - (last_sl or period_low)
            retracement = (close - (last_sl or period_low)) / (indication_range + 1e-12)
            if CORRECTION_MIN_PCT <= retracement <= CORRECTION_MAX_PCT:
                return "CORRECTION_BEAR"
            if retracement < CORRECTION_MIN_PCT:
                return "INDICATION_BEAR"

        return "NONE"

    if indication == "BULL":
        return "INDICATION_BULL"
    return "INDICATION_BEAR"


# ---------------------------------------------------------------------------
# Entry signal generator
# ---------------------------------------------------------------------------

def get_icc_entry(df: pd.DataFrame, pip_value: float = 0.0001) -> Optional[dict]:
    """
    Return an entry signal dict if a valid ICC continuation is detected.

    Scans the recent bars to find:
    1. An indication (new HH or LL)
    2. A correction (30–65% retrace)
    3. A continuation bar (close breaks back through correction extreme)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with at least 50 bars.  ``swing_high`` / ``swing_low``
        columns will be added if absent.
    pip_value : float
        One pip in price units (0.0001 for EUR/USD, 0.01 for USD/JPY, etc.).

    Returns
    -------
    dict or None
        Signal dict if setup is valid, else None.
        Keys: phase, entry, sl, correction_pct, indication_range_pips, aoi_nearby.
    """
    if len(df) < 30:
        logger.debug("Insufficient bars for ICC detection (%d)", len(df))
        return None

    if "swing_high" not in df.columns:
        df = detect_swing_points(df, lookback=5)

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n = len(df)

    # Scan backwards for the most recent indication → correction → continuation
    for lookback in range(10, min(60, n - 1)):
        window_end = n - 1
        window_start = max(0, window_end - lookback)
        hist_high = highs[window_start:window_end].max()
        hist_low  = lows[window_start:window_end].min()
        indication_range = hist_high - hist_low

        if indication_range < pip_value * 10:
            continue  # range too small to be meaningful

        cur_close = closes[window_end]

        # --- LONG setup ---
        # Indication: price previously broke above prior range
        # Correction: price pulled back 30–65% of that range
        # Continuation: current bar closes above correction low
        if cur_close > hist_low + indication_range * CORRECTION_MIN_PCT:
            # Find correction low (lowest low during pullback phase)
            correction_low = lows[max(0, window_end - 10): window_end].min()
            correction_pct = (hist_high - correction_low) / (indication_range + 1e-12)

            if CORRECTION_MIN_PCT <= correction_pct <= CORRECTION_MAX_PCT:
                if cur_close > correction_low:
                    # Valid CONTINUATION_LONG
                    entry = cur_close
                    sl    = correction_low - pip_value * 2  # 2 pip buffer below correction
                    range_pips = indication_range / pip_value

                    logger.info(
                        "ICC CONTINUATION_LONG detected: entry=%.5f sl=%.5f correction=%.1f%%",
                        entry, sl, correction_pct * 100,
                    )
                    return {
                        "phase": "CONTINUATION_LONG",
                        "entry": round(entry, 5),
                        "sl":    round(sl, 5),
                        "correction_pct": round(correction_pct * 100, 2),
                        "indication_range_pips": round(range_pips, 1),
                        "aoi_nearby": False,  # populated by SignalGenerator via AOIDetector
                    }

        # --- SHORT setup ---
        if cur_close < hist_high - indication_range * CORRECTION_MIN_PCT:
            correction_high = highs[max(0, window_end - 10): window_end].max()
            correction_pct  = (correction_high - hist_low) / (indication_range + 1e-12)

            if CORRECTION_MIN_PCT <= correction_pct <= CORRECTION_MAX_PCT:
                if cur_close < correction_high:
                    entry = cur_close
                    sl    = correction_high + pip_value * 2
                    range_pips = indication_range / pip_value

                    logger.info(
                        "ICC CONTINUATION_SHORT detected: entry=%.5f sl=%.5f correction=%.1f%%",
                        entry, sl, correction_pct * 100,
                    )
                    return {
                        "phase": "CONTINUATION_SHORT",
                        "entry": round(entry, 5),
                        "sl":    round(sl, 5),
                        "correction_pct": round(correction_pct * 100, 2),
                        "indication_range_pips": round(range_pips, 1),
                        "aoi_nearby": False,
                    }

    return None


# ---------------------------------------------------------------------------
# AOI (Areas of Interest) detector
# ---------------------------------------------------------------------------

class AOIDetector:
    """
    Areas of Interest (AOI) detector — daily and weekly price extremes.

    Used to flag whether a proposed entry price is near a significant
    structural level (daily high/low, weekly high/low), which increases
    the probability of a meaningful reaction.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily OHLCV DataFrame (columns: datetime/date, open, high, low, close).
    """

    def __init__(self, df_daily: pd.DataFrame) -> None:
        if df_daily.empty:
            self._levels: list[dict] = []
            return

        daily = df_daily.copy()
        if "datetime" in daily.columns:
            daily["date"] = pd.to_datetime(daily["datetime"]).dt.date
        elif "date" not in daily.columns:
            raise ValueError("df_daily must have a 'datetime' or 'date' column")

        daily["date"] = pd.to_datetime(daily["date"])

        self._levels = self._extract_levels(daily)
        logger.info("AOIDetector initialised with %d levels", len(self._levels))

    def _extract_levels(self, daily: pd.DataFrame) -> list[dict]:
        """Build AOI level list from daily OHLCV."""
        levels: list[dict] = []

        # Last 5 daily high/low extremes
        for _, row in daily.tail(5).iterrows():
            levels.append({"price": row["high"], "type": "daily_high", "date": row["date"]})
            levels.append({"price": row["low"],  "type": "daily_low",  "date": row["date"]})

        # Weekly high/low — resample
        weekly = daily.resample("W", on="date").agg(
            weekly_high=("high", "max"),
            weekly_low=("low", "min"),
        ).tail(4)

        for dt, row in weekly.iterrows():
            levels.append({"price": row["weekly_high"], "type": "weekly_high", "date": dt})
            levels.append({"price": row["weekly_low"],  "type": "weekly_low",  "date": dt})

        return levels

    def get_nearby_aoi(
        self,
        price: float,
        tolerance_pips: float = AOI_TOLERANCE_PIPS_DEFAULT,
        pip_value: float = 0.0001,
    ) -> list[dict]:
        """
        Return all AOI levels within ``tolerance_pips`` of ``price``.

        Parameters
        ----------
        price : float
            Current or entry price.
        tolerance_pips : float
            Distance threshold in pips.
        pip_value : float
            One pip in price units.

        Returns
        -------
        list[dict]
            Each dict has keys: price, type, date.
        """
        threshold = tolerance_pips * pip_value
        return [lvl for lvl in self._levels if abs(lvl["price"] - price) <= threshold]

    def is_at_aoi(
        self,
        price: float,
        tolerance_pips: float = AOI_TOLERANCE_PIPS_DEFAULT,
        pip_value: float = 0.0001,
    ) -> bool:
        """
        Return True if ``price`` is within ``tolerance_pips`` of any AOI.

        Parameters
        ----------
        price : float
            Price to check.
        tolerance_pips : float
            Distance threshold in pips.
        pip_value : float
            One pip in price units.

        Returns
        -------
        bool
        """
        return len(self.get_nearby_aoi(price, tolerance_pips, pip_value)) > 0
