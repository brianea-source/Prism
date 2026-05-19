"""Session-level Power of 3 bias engine.

Implements the ICT daily cycle:
  Asian session  → ACCUMULATION  (range forms, mark high/low)
  London session → MANIPULATION  (sweep of Asian high or low)
  NY session     → DISTRIBUTION  (real move, opposite to sweep)

Direction comes from the sweep:
  HIGH swept (buyside liquidity grabbed) → distribution goes SHORT
  LOW  swept (sellside liquidity grabbed) → distribution goes LONG

This module replaces HTF swing alignment as the *primary* direction-setter.
HTF bias is demoted to a confidence modifier, not a gate.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd

from prism.delivery.session_filter import asian_session_bars
from prism.execution.mt5_bridge import PIP_SIZE

logger = logging.getLogger(__name__)


class SessionPhase(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    DISTRIBUTION = "DISTRIBUTION"
    NO_DATA = "NO_DATA"


@dataclass
class AsianRange:
    """The accumulation range from 00:00–06:00 UTC."""
    high: float
    low: float
    midpoint: float
    range_pips: float
    bar_count: int
    date: date


@dataclass
class SessionBias:
    """Result of the session-level Po3 analysis."""
    direction: Optional[str]
    phase: SessionPhase
    asian_range: Optional[AsianRange]
    sweep_side: Optional[str]
    sweep_price: Optional[float]
    sweep_confirmed: bool
    displacement_confirmed: bool
    confidence_modifier: float


def _pip(instrument: str) -> float:
    return float(PIP_SIZE.get(instrument, 0.0001))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class SessionBiasEngine:
    """Determines today's trade direction from the Asian-to-London Po3 cycle.

    Lifecycle per trading day:
    1. ``load_asian_range(m5_bars)`` — called once on first London scan.
       Extracts the 00:00–06:00 UTC range from the M5 history the runner
       already fetches (500 bars cover ~41 hours, always includes Asian).
    2. ``update(current_bars)`` — called every scan during London/NY.
       Checks whether price has swept the Asian high or low and whether
       displacement has followed.
    3. ``bias`` property — current session direction, or None if no sweep yet.
    """

    def __init__(self, instrument: str) -> None:
        self.instrument = instrument
        self._asian_range: Optional[AsianRange] = None
        self._sweep_side: Optional[str] = None
        self._sweep_price: Optional[float] = None
        self._sweep_confirmed = False
        self._displacement_confirmed = False
        self._today: Optional[date] = None

    def _reset_day(self, today: date) -> None:
        self._asian_range = None
        self._sweep_side = None
        self._sweep_price = None
        self._sweep_confirmed = False
        self._displacement_confirmed = False
        self._today = today

    @property
    def asian_range(self) -> Optional[AsianRange]:
        return self._asian_range

    def load_asian_range(
        self, m5_bars: pd.DataFrame, ref_date: date = None,
    ) -> Optional[AsianRange]:
        """Extract today's Asian range from the M5 bar history.

        Returns None if fewer than ``PRISM_MIN_ASIAN_BARS`` (default 6 = 30 min)
        bars fall within the Asian window. A tight filter: we need enough bars
        to form a meaningful range.
        """
        if ref_date is None:
            ref_date = datetime.now(timezone.utc).date()

        if self._today != ref_date:
            self._reset_day(ref_date)

        asian_bars = asian_session_bars(m5_bars, ref_date)
        min_bars = _env_int("PRISM_MIN_ASIAN_BARS", 6)

        if len(asian_bars) < min_bars:
            logger.debug(
                "Asian range: only %d bars (need %d) — no range today",
                len(asian_bars), min_bars,
            )
            return None

        pip = _pip(self.instrument)
        high = float(asian_bars["high"].max())
        low = float(asian_bars["low"].min())
        self._asian_range = AsianRange(
            high=high,
            low=low,
            midpoint=round((high + low) / 2, 5),
            range_pips=round((high - low) / pip, 2),
            bar_count=len(asian_bars),
            date=ref_date,
        )
        logger.info(
            "Asian range loaded: %.5f–%.5f (%.1f pips, %d bars)",
            low, high, self._asian_range.range_pips, self._asian_range.bar_count,
        )
        return self._asian_range

    def update(self, current_bars: pd.DataFrame) -> SessionBias:
        """Check whether the manipulation sweep has occurred and displacement started.

        ``current_bars`` is the live M5 feed (the same ``entry_df`` the runner
        already passes to the generator). We scan bars that arrived after the
        Asian session close (06:00 UTC) for sweep + displacement.
        """
        if self._asian_range is None:
            return SessionBias(
                direction=None,
                phase=SessionPhase.NO_DATA,
                asian_range=None,
                sweep_side=None,
                sweep_price=None,
                sweep_confirmed=False,
                displacement_confirmed=False,
                confidence_modifier=0.0,
            )

        ar = self._asian_range
        pip = _pip(self.instrument)
        min_disp_pips = _env_float("PRISM_SESSION_DISPLACEMENT_PIPS", 15.0)
        threshold = min_disp_pips * pip

        # Filter to post-Asian bars (after 06:00 UTC today)
        post_asian = self._post_asian_bars(current_bars)
        if post_asian.empty:
            return self._build_bias()

        # --- Sweep detection ---
        if not self._sweep_confirmed:
            for _, row in post_asian.iterrows():
                hi = float(row["high"])
                lo = float(row["low"])
                cl = float(row["close"])

                # HIGH sweep: wick above Asian high, close back below
                if hi > ar.high and cl <= ar.high:
                    self._sweep_side = "HIGH"
                    self._sweep_price = hi
                    self._sweep_confirmed = True
                    logger.info(
                        "Session Po3: HIGH swept at %.5f (Asian high=%.5f) — "
                        "bias = SHORT",
                        hi, ar.high,
                    )
                    break

                # LOW sweep: wick below Asian low, close back above
                if lo < ar.low and cl >= ar.low:
                    self._sweep_side = "LOW"
                    self._sweep_price = lo
                    self._sweep_confirmed = True
                    logger.info(
                        "Session Po3: LOW swept at %.5f (Asian low=%.5f) — "
                        "bias = LONG",
                        lo, ar.low,
                    )
                    break

        # --- Displacement detection (after sweep) ---
        if self._sweep_confirmed and not self._displacement_confirmed:
            disp_bars = _env_int("PRISM_SESSION_DISPLACEMENT_BARS", 12)
            sweep_idx = self._find_sweep_bar_index(post_asian)
            if sweep_idx is not None:
                end_idx = min(len(post_asian), sweep_idx + 1 + disp_bars)
                for j in range(sweep_idx + 1, end_idx):
                    row = post_asian.iloc[j]
                    if self._sweep_side == "HIGH":
                        if self._sweep_price is not None and \
                                self._sweep_price - float(row["low"]) >= threshold:
                            self._displacement_confirmed = True
                            logger.info(
                                "Session Po3: SHORT displacement confirmed "
                                "(%.1f pips from sweep)",
                                (self._sweep_price - float(row["low"])) / pip,
                            )
                            break
                    elif self._sweep_side == "LOW":
                        if self._sweep_price is not None and \
                                float(row["high"]) - self._sweep_price >= threshold:
                            self._displacement_confirmed = True
                            logger.info(
                                "Session Po3: LONG displacement confirmed "
                                "(%.1f pips from sweep)",
                                (float(row["high"]) - self._sweep_price) / pip,
                            )
                            break

        return self._build_bias()

    def _post_asian_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to bars after 06:00 UTC on the Asian range date."""
        if self._asian_range is None:
            return df.iloc[0:0]
        d = self._asian_range.date
        cutoff = datetime(d.year, d.month, d.day, 6, 0, tzinfo=timezone.utc)

        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index.tz_convert(timezone.utc) if df.index.tz is not None \
                else df.index.tz_localize(timezone.utc)
            return df.loc[ts >= cutoff]
        elif "datetime" in df.columns:
            col = pd.to_datetime(df["datetime"], utc=True)
            return df.loc[col >= cutoff]
        return df

    def _find_sweep_bar_index(self, post_asian: pd.DataFrame) -> Optional[int]:
        """Index (iloc position) of the bar that executed the sweep."""
        if not self._sweep_confirmed or self._asian_range is None:
            return None
        ar = self._asian_range
        for i, (_, row) in enumerate(post_asian.iterrows()):
            hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
            if self._sweep_side == "HIGH" and hi > ar.high and cl <= ar.high:
                return i
            if self._sweep_side == "LOW" and lo < ar.low and cl >= ar.low:
                return i
        return None

    def _build_bias(self) -> SessionBias:
        direction: Optional[str] = None
        if self._sweep_confirmed:
            direction = "SHORT" if self._sweep_side == "HIGH" else "LONG"

        if not self._sweep_confirmed:
            phase = SessionPhase.ACCUMULATION
        elif not self._displacement_confirmed:
            phase = SessionPhase.MANIPULATION
        else:
            phase = SessionPhase.DISTRIBUTION

        # Confidence modifier: clean sweep + displacement = 1.0
        modifier = 0.0
        if self._sweep_confirmed:
            modifier = 0.6
        if self._displacement_confirmed:
            modifier = 1.0

        return SessionBias(
            direction=direction,
            phase=phase,
            asian_range=self._asian_range,
            sweep_side=self._sweep_side,
            sweep_price=self._sweep_price,
            sweep_confirmed=self._sweep_confirmed,
            displacement_confirmed=self._displacement_confirmed,
            confidence_modifier=modifier,
        )
