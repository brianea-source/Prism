"""Liquidity Sweep Detection (PRISM Phase 6.C).

A *sweep* is when price wicks above a recent swing high (or below a swing low)
and then closes back inside — i.e. it took out resting stops without committing
to the breakout. In ICT/SMC parlance, this is the *manipulation* leg that
precedes the real distribution move:

* ``HIGH_SWEEP``  → price grabbed sell-side liquidity above the highs and
  reversed; supports a SHORT entry.
* ``LOW_SWEEP``   → price grabbed buy-side liquidity below the lows and
  reversed; supports a LONG entry.

This module is pure detection: no trading decisions, no env reads.
``SignalGenerator`` consumes results via ``has_recent_sweep``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from prism.execution.mt5_bridge import PIP_SIZE


@dataclass
class LiquiditySweep:
    """A single confirmed liquidity sweep event."""

    instrument: str
    type: str  # "HIGH_SWEEP" or "LOW_SWEEP"
    swept_level: float
    sweep_bar: int
    close_inside: bool
    timestamp: str
    displacement_followed: bool


# Pips of follow-through after the sweep that count as displacement.
DEFAULT_DISPLACEMENT_PIPS = 5.0
# How many bars after the sweep we look for displacement.
DEFAULT_DISPLACEMENT_LOOKAHEAD = 3


class SweepDetector:
    """Detects liquidity sweeps over a rolling lookback window."""

    def __init__(self, instrument: str, lookback: int = 20) -> None:
        self.instrument = instrument
        self.lookback = int(lookback)
        self.sweeps: list[LiquiditySweep] = []
        # Index of the last bar seen by ``detect()`` — anchors ``has_recent_sweep``.
        self._latest_scanned_bar: Optional[int] = None

    @property
    def latest_scanned_bar(self) -> Optional[int]:
        """Public accessor for the last bar index seen by :meth:`detect`."""
        return self._latest_scanned_bar

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

    def _bar_timestamp(self, df: pd.DataFrame, idx: int) -> str:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[idx]
            if hasattr(ts, "isoformat"):
                return ts.isoformat()
            return str(ts)
        return f"bar-{idx}"

    def _displacement_after(
        self,
        df: pd.DataFrame,
        sweep_idx: int,
        sweep_type: str,
        min_pips: float,
        lookahead: int,
    ) -> bool:
        """Did price travel ``min_pips`` against the swept side within ``lookahead``?"""
        pip = self._pip_size()
        threshold = min_pips * pip
        end = min(len(df), sweep_idx + 1 + lookahead)
        sweep_close = float(df.iloc[sweep_idx]["close"])
        for j in range(sweep_idx + 1, end):
            row = df.iloc[j]
            if sweep_type == "HIGH_SWEEP":
                # Reversal goes DOWN: low must drop ``threshold`` below sweep close.
                if sweep_close - float(row["low"]) >= threshold:
                    return True
            else:  # LOW_SWEEP
                if float(row["high"]) - sweep_close >= threshold:
                    return True
        return False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def detect(
        self,
        df: pd.DataFrame,
        min_displacement_pips: float = DEFAULT_DISPLACEMENT_PIPS,
        lookahead: int = DEFAULT_DISPLACEMENT_LOOKAHEAD,
    ) -> list[LiquiditySweep]:
        """Scan ``df`` and return all sweeps found.

        - ``HIGH_SWEEP``: ``high[i] > max(high[i-lookback:i])`` AND
          ``close[i] < that level``.
        - ``LOW_SWEEP``: mirror.
        - ``displacement_followed`` is True when, within ``lookahead`` bars after
          the sweep, price travels at least ``min_displacement_pips`` against the
          swept side (measured from the sweep bar's close).

        Idempotent: re-running ``detect`` on the same df won't duplicate (sweeps
        already in ``self.sweeps`` for the same ``sweep_bar`` + ``type`` are skipped).
        """
        if df.empty:
            return []
        self._require_ohlc(df)
        if len(df) <= self.lookback:
            return []

        seen: set[tuple[str, int]] = {(s.type, s.sweep_bar) for s in self.sweeps}
        new_sweeps: list[LiquiditySweep] = []

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()

        for i in range(self.lookback, len(df)):
            window_hi = float(highs[i - self.lookback : i].max())
            window_lo = float(lows[i - self.lookback : i].min())
            hi_i = float(highs[i])
            lo_i = float(lows[i])
            cl_i = float(closes[i])

            # --- HIGH_SWEEP ------------------------------------------------
            if hi_i > window_hi and cl_i < window_hi:
                key = ("HIGH_SWEEP", i)
                if key not in seen:
                    seen.add(key)
                    sweep = LiquiditySweep(
                        instrument=self.instrument,
                        type="HIGH_SWEEP",
                        swept_level=window_hi,
                        sweep_bar=int(i),
                        close_inside=True,  # close < swept level by construction
                        timestamp=self._bar_timestamp(df, i),
                        displacement_followed=self._displacement_after(
                            df, i, "HIGH_SWEEP", min_displacement_pips, lookahead
                        ),
                    )
                    self.sweeps.append(sweep)
                    new_sweeps.append(sweep)

            # --- LOW_SWEEP -------------------------------------------------
            if lo_i < window_lo and cl_i > window_lo:
                key = ("LOW_SWEEP", i)
                if key not in seen:
                    seen.add(key)
                    sweep = LiquiditySweep(
                        instrument=self.instrument,
                        type="LOW_SWEEP",
                        swept_level=window_lo,
                        sweep_bar=int(i),
                        close_inside=True,
                        timestamp=self._bar_timestamp(df, i),
                        displacement_followed=self._displacement_after(
                            df, i, "LOW_SWEEP", min_displacement_pips, lookahead
                        ),
                    )
                    self.sweeps.append(sweep)
                    new_sweeps.append(sweep)

        self._latest_scanned_bar = len(df) - 1
        return new_sweeps

    @staticmethod
    def _required_type(direction: str) -> Optional[str]:
        d = (direction or "").strip().upper()
        if d in {"LONG", "BUY", "BULLISH"}:
            return "LOW_SWEEP"
        if d in {"SHORT", "SELL", "BEARISH"}:
            return "HIGH_SWEEP"
        return None

    def has_recent_sweep(
        self,
        direction: str,
        bars_back: int = 5,
        require_displacement: bool = True,
    ) -> bool:
        """Is there a sweep matching ``direction`` within the last ``bars_back`` bars?

        ``LONG`` → looks for ``LOW_SWEEP`` (manipulation took lows, ready to push up).
        ``SHORT`` → looks for ``HIGH_SWEEP``.

        When ``require_displacement`` is True (default), only sweeps whose
        ``displacement_followed`` is True qualify — this filters out sweeps that
        didn't actually start the distribution leg.
        """
        want = self._required_type(direction)
        if want is None or not self.sweeps:
            return False
        # Anchor "recent" to the last bar scanned, falling back to the latest
        # sweep itself when ``detect()`` hasn't been called yet.
        anchor = (
            self._latest_scanned_bar
            if self._latest_scanned_bar is not None
            else max(s.sweep_bar for s in self.sweeps)
        )
        cutoff = anchor - int(bars_back)
        for s in self.sweeps:
            if s.type != want:
                continue
            if s.sweep_bar < cutoff:
                continue
            if require_displacement and not s.displacement_followed:
                continue
            return True
        return False

    def last_sweep(self, direction: str) -> Optional[LiquiditySweep]:
        """Most recent sweep (by ``sweep_bar``) matching ``direction``, or None."""
        want = self._required_type(direction)
        if want is None:
            return None
        matching = [s for s in self.sweeps if s.type == want]
        if not matching:
            return None
        return max(matching, key=lambda s: s.sweep_bar)
