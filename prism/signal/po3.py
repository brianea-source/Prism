"""Power of Three (Po3) phase detection (PRISM Phase 6.C).

Each kill-zone session is modelled as three behavioural phases, in order:

1. **ACCUMULATION** — price chops in a tight range as smart-money fills orders.
2. **MANIPULATION** — a sweep takes out the accumulation high or low,
   triggering retail stops.
3. **DISTRIBUTION** — once the sweep is done, price travels in the *opposite*
   direction of the wick, delivering the real session move.

PRISM only enters in DISTRIBUTION (sweep + displacement both true). This module
classifies the current state of a session given an OHLC window; downstream
components in ``SignalGenerator`` gate on ``Po3Detector.is_entry_phase``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from prism.execution.mt5_bridge import PIP_SIZE


class Po3Phase(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    DISTRIBUTION = "DISTRIBUTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class Po3State:
    phase: Po3Phase
    session: str
    session_open: float
    session_high: float
    session_low: float
    range_size_pips: float
    sweep_detected: bool
    displacement_detected: bool


# Defaults: M5 bars, retail-friendly thresholds. Override per-call when needed.
DEFAULT_ACCUM_BARS = 12  # ~60 min of M5
DEFAULT_DISPLACEMENT_PIPS = 15.0


class Po3Detector:
    """Classify the current Po3 phase from a session's OHLC window."""

    def __init__(self, instrument: str) -> None:
        self.instrument = instrument

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _pip_size(self) -> float:
        return float(PIP_SIZE.get(self.instrument, 0.0001))

    @staticmethod
    def _require_ohlc(df: pd.DataFrame) -> None:
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

    @staticmethod
    def _empty_state(session: str) -> Po3State:
        return Po3State(
            phase=Po3Phase.UNKNOWN,
            session=session,
            session_open=float("nan"),
            session_high=float("nan"),
            session_low=float("nan"),
            range_size_pips=0.0,
            sweep_detected=False,
            displacement_detected=False,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def detect_phase(
        self,
        df: pd.DataFrame,
        session: str,
        accumulation_bars: int = DEFAULT_ACCUM_BARS,
        displacement_pips: float = DEFAULT_DISPLACEMENT_PIPS,
    ) -> Po3State:
        """Classify the active Po3 phase across the given session window.

        ``df`` is expected to be the bars belonging to the named ``session``
        (caller pre-filters by kill zone). The first ``accumulation_bars`` bars
        define the accumulation high / low.
        """
        if df is None or df.empty:
            return self._empty_state(session)
        self._require_ohlc(df)

        pip = self._pip_size()
        session_open = float(df.iloc[0]["open"])
        session_high = float(df["high"].max())
        session_low = float(df["low"].min())
        range_size_pips = round((session_high - session_low) / pip, 4)

        # Not enough data to declare anything past ACCUMULATION yet.
        if len(df) <= accumulation_bars:
            return Po3State(
                phase=Po3Phase.ACCUMULATION,
                session=session,
                session_open=session_open,
                session_high=session_high,
                session_low=session_low,
                range_size_pips=range_size_pips,
                sweep_detected=False,
                displacement_detected=False,
            )

        accum = df.iloc[:accumulation_bars]
        accum_high = float(accum["high"].max())
        accum_low = float(accum["low"].min())
        post = df.iloc[accumulation_bars:]

        # --- sweep detection ------------------------------------------------
        sweep_idx: Optional[int] = None
        sweep_side: Optional[str] = None
        for j, (_, row) in enumerate(post.iterrows()):
            hi = float(row["high"])
            lo = float(row["low"])
            cl = float(row["close"])
            if hi > accum_high and cl < accum_high:
                sweep_idx = accumulation_bars + j
                sweep_side = "HIGH"
                break
            if lo < accum_low and cl > accum_low:
                sweep_idx = accumulation_bars + j
                sweep_side = "LOW"
                break

        sweep_detected = sweep_idx is not None

        # --- displacement detection (after sweep, opposite side) ----------
        displacement_detected = False
        if sweep_detected:
            sweep_close = float(df.iloc[sweep_idx]["close"])
            threshold = displacement_pips * pip
            for k in range(sweep_idx + 1, len(df)):
                row = df.iloc[k]
                if sweep_side == "HIGH":
                    if sweep_close - float(row["low"]) >= threshold:
                        displacement_detected = True
                        break
                else:  # LOW sweep → distribution moves UP
                    if float(row["high"]) - sweep_close >= threshold:
                        displacement_detected = True
                        break

        if sweep_detected and displacement_detected:
            phase = Po3Phase.DISTRIBUTION
        elif sweep_detected:
            phase = Po3Phase.MANIPULATION
        else:
            phase = Po3Phase.ACCUMULATION

        return Po3State(
            phase=phase,
            session=session,
            session_open=session_open,
            session_high=session_high,
            session_low=session_low,
            range_size_pips=range_size_pips,
            sweep_detected=sweep_detected,
            displacement_detected=displacement_detected,
        )

    def is_entry_phase(self, state: Po3State) -> bool:
        """True only when manipulation has completed *and* distribution started."""
        return bool(state.sweep_detected and state.displacement_detected)
