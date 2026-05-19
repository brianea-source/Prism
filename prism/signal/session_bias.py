"""Session-level Power of 3 bias engine.

Implements the ICT intra-day cycle:

  **London kill zone (07:00–11:00 UTC):**
    Accumulation = Asian session (00:00–06:00 UTC)
    Manipulation = London sweeps Asian high or low
    Distribution = real move, opposite to sweep

  **NY kill zone (13:00–17:00 UTC):**
    Accumulation = London session (07:00–11:00 UTC)
    Manipulation = NY sweeps London high or low
    Distribution = real move, opposite to sweep

Each kill zone has its OWN Po3 cycle with its OWN reference range.
Direction comes from the sweep:
  HIGH swept (buyside liquidity grabbed) → distribution goes SHORT
  LOW  swept (sellside liquidity grabbed) → distribution goes LONG
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd

from prism.execution.mt5_bridge import PIP_SIZE

logger = logging.getLogger(__name__)


class SessionPhase(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    DISTRIBUTION = "DISTRIBUTION"
    NO_DATA = "NO_DATA"


@dataclass
class AccumulationRange:
    """The reference range from the prior session (accumulation)."""
    high: float
    low: float
    midpoint: float
    range_pips: float
    bar_count: int
    source_session: str  # "ASIAN" or "LONDON"
    date: date


# Keep the old name as an alias for backward compat in tests/generator
AsianRange = AccumulationRange


@dataclass
class SessionBias:
    """Result of the session-level Po3 analysis."""
    direction: Optional[str]
    phase: SessionPhase
    asian_range: Optional[AccumulationRange]  # named for compat; holds any accum range
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


# ── Session time windows (UTC) ───────────────────────────────────────────
ASIAN_START_HOUR, ASIAN_END_HOUR = 0, 6
LONDON_START_HOUR, LONDON_END_HOUR = 7, 11
NY_START_HOUR, NY_END_HOUR = 13, 17


def _filter_bars_by_hour_range(
    df: pd.DataFrame, ref_date: date, start_hour: int, end_hour: int,
) -> pd.DataFrame:
    """Return rows whose timestamp falls within [start_hour, end_hour) UTC on ``ref_date``."""
    start = datetime(ref_date.year, ref_date.month, ref_date.day,
                     start_hour, 0, tzinfo=timezone.utc)
    end = datetime(ref_date.year, ref_date.month, ref_date.day,
                   end_hour, 0, tzinfo=timezone.utc)

    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index.tz_convert(timezone.utc) if df.index.tz is not None \
            else df.index.tz_localize(timezone.utc)
        mask = (ts >= start) & (ts < end)
    elif "datetime" in df.columns:
        col = pd.to_datetime(df["datetime"], utc=True)
        mask = (col >= start) & (col < end)
    else:
        return df.iloc[0:0]
    return df.loc[mask]


def _filter_bars_after(
    df: pd.DataFrame, ref_date: date, after_hour: int,
) -> pd.DataFrame:
    """Return rows whose timestamp is >= ``after_hour`` UTC on ``ref_date``."""
    cutoff = datetime(ref_date.year, ref_date.month, ref_date.day,
                      after_hour, 0, tzinfo=timezone.utc)
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index.tz_convert(timezone.utc) if df.index.tz is not None \
            else df.index.tz_localize(timezone.utc)
        return df.loc[ts >= cutoff]
    elif "datetime" in df.columns:
        col = pd.to_datetime(df["datetime"], utc=True)
        return df.loc[col >= cutoff]
    return df


class SessionBiasEngine:
    """Determines trade direction from the prior-session sweep.

    Works for both kill zones:
    - London: loads Asian range, detects London sweep of Asian H/L
    - NY: loads London range, detects NY sweep of London H/L

    The engine auto-detects which reference range to use based on the
    current UTC hour.

    Lifecycle per scan:
    1. ``load_accumulation_range(m5_bars)`` — extracts the prior session range
    2. ``update(current_bars)`` — checks for sweep + displacement
    3. Result via ``SessionBias`` — direction, phase, confidence modifier
    """

    def __init__(self, instrument: str) -> None:
        self.instrument = instrument
        self._accum_range: Optional[AccumulationRange] = None
        self._sweep_side: Optional[str] = None
        self._sweep_price: Optional[float] = None
        self._sweep_confirmed = False
        self._displacement_confirmed = False
        self._today: Optional[date] = None
        self._active_session: Optional[str] = None

    def _reset(self, today: date, session: str) -> None:
        self._accum_range = None
        self._sweep_side = None
        self._sweep_price = None
        self._sweep_confirmed = False
        self._displacement_confirmed = False
        self._today = today
        self._active_session = session

    @property
    def asian_range(self) -> Optional[AccumulationRange]:
        """Backward-compatible accessor."""
        return self._accum_range

    @property
    def accumulation_range(self) -> Optional[AccumulationRange]:
        return self._accum_range

    def _detect_session(self) -> str:
        """Return 'LONDON' or 'NY' based on current UTC hour."""
        hour = datetime.now(timezone.utc).hour
        if LONDON_START_HOUR <= hour < LONDON_END_HOUR:
            return "LONDON"
        if NY_START_HOUR <= hour < NY_END_HOUR:
            return "NY"
        # Between London close and NY open, or off-hours — use NY if
        # we're past London, otherwise default to London.
        if hour >= LONDON_END_HOUR:
            return "NY"
        return "LONDON"

    def load_accumulation_range(
        self, m5_bars: pd.DataFrame, ref_date: date = None,
        force_session: str = None,
    ) -> Optional[AccumulationRange]:
        """Extract the prior session's range as the accumulation reference.

        For London scans: extracts Asian range (00:00–06:00 UTC)
        For NY scans: extracts London range (07:00–11:00 UTC)

        ``force_session`` overrides auto-detection (for testing).
        """
        if ref_date is None:
            ref_date = datetime.now(timezone.utc).date()

        session = force_session or self._detect_session()

        # Reset if day changed or session changed (London → NY transition)
        if self._today != ref_date or self._active_session != session:
            self._reset(ref_date, session)

        if session == "LONDON":
            accum_bars = _filter_bars_by_hour_range(
                m5_bars, ref_date, ASIAN_START_HOUR, ASIAN_END_HOUR,
            )
            source = "ASIAN"
            range_label = "Asian"
        else:
            accum_bars = _filter_bars_by_hour_range(
                m5_bars, ref_date, LONDON_START_HOUR, LONDON_END_HOUR,
            )
            source = "LONDON"
            range_label = "London"

        min_bars = _env_int("PRISM_MIN_ASIAN_BARS", 6)
        if len(accum_bars) < min_bars:
            logger.debug(
                "%s range: only %d bars (need %d) — no range",
                range_label, len(accum_bars), min_bars,
            )
            return None

        pip = _pip(self.instrument)
        high = float(accum_bars["high"].max())
        low = float(accum_bars["low"].min())
        self._accum_range = AccumulationRange(
            high=high,
            low=low,
            midpoint=round((high + low) / 2, 5),
            range_pips=round((high - low) / pip, 2),
            bar_count=len(accum_bars),
            source_session=source,
            date=ref_date,
        )
        logger.info(
            "%s range loaded: %.5f–%.5f (%.1f pips, %d bars)",
            range_label, low, high,
            self._accum_range.range_pips, self._accum_range.bar_count,
        )
        return self._accum_range

    # Keep old name as alias
    def load_asian_range(
        self, m5_bars: pd.DataFrame, ref_date: date = None,
    ) -> Optional[AccumulationRange]:
        """Backward-compatible alias for ``load_accumulation_range``."""
        return self.load_accumulation_range(m5_bars, ref_date=ref_date)

    def update(self, current_bars: pd.DataFrame) -> SessionBias:
        """Check whether the manipulation sweep has occurred and displacement started.

        Scans bars after the accumulation session close for sweep + displacement.
        """
        if self._accum_range is None:
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

        ar = self._accum_range
        pip = _pip(self.instrument)
        min_disp_pips = _env_float("PRISM_SESSION_DISPLACEMENT_PIPS", 15.0)
        threshold = min_disp_pips * pip

        # Bars after the accumulation session closed
        if ar.source_session == "ASIAN":
            after_hour = ASIAN_END_HOUR
        else:
            after_hour = LONDON_END_HOUR
        post_accum = _filter_bars_after(current_bars, ar.date, after_hour)
        if post_accum.empty:
            return self._build_bias()

        # --- Sweep detection ---
        if not self._sweep_confirmed:
            for _, row in post_accum.iterrows():
                hi = float(row["high"])
                lo = float(row["low"])
                cl = float(row["close"])

                if hi > ar.high and cl <= ar.high:
                    self._sweep_side = "HIGH"
                    self._sweep_price = hi
                    self._sweep_confirmed = True
                    logger.info(
                        "Session Po3 [%s]: HIGH swept at %.5f "
                        "(%s high=%.5f) — bias = SHORT",
                        self._active_session, hi, ar.source_session, ar.high,
                    )
                    break

                if lo < ar.low and cl >= ar.low:
                    self._sweep_side = "LOW"
                    self._sweep_price = lo
                    self._sweep_confirmed = True
                    logger.info(
                        "Session Po3 [%s]: LOW swept at %.5f "
                        "(%s low=%.5f) — bias = LONG",
                        self._active_session, lo, ar.source_session, ar.low,
                    )
                    break

        # --- Displacement detection ---
        if self._sweep_confirmed and not self._displacement_confirmed:
            disp_bars = _env_int("PRISM_SESSION_DISPLACEMENT_BARS", 12)
            sweep_idx = self._find_sweep_bar_index(post_accum)
            if sweep_idx is not None:
                end_idx = min(len(post_accum), sweep_idx + 1 + disp_bars)
                for j in range(sweep_idx + 1, end_idx):
                    row = post_accum.iloc[j]
                    if self._sweep_side == "HIGH":
                        if self._sweep_price is not None and \
                                self._sweep_price - float(row["low"]) >= threshold:
                            self._displacement_confirmed = True
                            logger.info(
                                "Session Po3 [%s]: SHORT displacement "
                                "(%.1f pips)",
                                self._active_session,
                                (self._sweep_price - float(row["low"])) / pip,
                            )
                            break
                    elif self._sweep_side == "LOW":
                        if self._sweep_price is not None and \
                                float(row["high"]) - self._sweep_price >= threshold:
                            self._displacement_confirmed = True
                            logger.info(
                                "Session Po3 [%s]: LONG displacement "
                                "(%.1f pips)",
                                self._active_session,
                                (float(row["high"]) - self._sweep_price) / pip,
                            )
                            break

        return self._build_bias()

    def _find_sweep_bar_index(self, post_accum: pd.DataFrame) -> Optional[int]:
        """iloc index of the bar that executed the sweep."""
        if not self._sweep_confirmed or self._accum_range is None:
            return None
        ar = self._accum_range
        for i, (_, row) in enumerate(post_accum.iterrows()):
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

        modifier = 0.0
        if self._sweep_confirmed:
            modifier = 0.6
        if self._displacement_confirmed:
            modifier = 1.0

        return SessionBias(
            direction=direction,
            phase=phase,
            asian_range=self._accum_range,
            sweep_side=self._sweep_side,
            sweep_price=self._sweep_price,
            sweep_confirmed=self._sweep_confirmed,
            displacement_confirmed=self._displacement_confirmed,
            confidence_modifier=modifier,
        )
