"""
Phase 8 — Trade Quality Filter (Task 1.1, Stage 1).

This module implements PRD2 §8 helpers + a thin Stage 1 gate that the signal
generator can run AFTER Layer 3 (FVG entry) and BEFORE the final
``SignalPacket`` assembly. In Stage 1 the gate is wired in fail-soft mode and
defaults to OFF (``PRISM_QUALITY_FILTER_ENABLED=0``) so the live runner's
behavior is unchanged. Operators can flip it on per environment to start
shadow-evaluating signals against the quality criteria.

Public surface (PRD2 §8 / ACTION_PLAN.md Task 1.1):

* :func:`check_ote_zone` — Optimal Trade Entry fib check (62-79% retracement).
* :func:`score_fvg_quality` — multi-factor 0.0-1.0 quality score for an FVG.
* :func:`check_candle_confirmation` — reversal candle + 8 EMA cross.
* :func:`next_liquidity_pool` — dynamic R:R target from unconsumed swings.
* :func:`apply_quality_filter` — orchestrator used by the generator. Returns
  ``QualityCheckResult`` and is responsible for the env-flag short-circuit.

Stage 1 design constraints:

* The full ``QualityFilter`` class shown in PRD2 §8 is deferred to Stage 2
  (Task 1.3 will need the score for ML features anyway). Stage 1 ships the
  primitives and a single ``apply_quality_filter`` entry point so the wiring
  surface stays minimal.
* No change to ``MIN_RR`` (1.5 → 2.0). PRD2 calls out a separate fixture audit
  for that promotion; out of scope for this PR.
* Fail-soft: a quality check raising is logged and treated as a pass when the
  flag is off, and as a block when the flag is on. The generator never crashes
  on a bad quality signal.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables — exposed as env vars so ops can A/B without redeploys.
# ---------------------------------------------------------------------------

# Optimal Trade Entry zone (ICT). 0.62-0.79 fib retracement of the last leg.
OTE_LOWER_FIB = 0.62
OTE_UPPER_FIB = 0.79

# Default FVG quality threshold. Calibrated for Phase 8 Stage 1: anything below
# 0.5 is treated as a low-quality gap. Will be re-tuned after Phase 7.B walk-
# forward results are in.
DEFAULT_MIN_FVG_QUALITY = 0.5

# Component weights for ``score_fvg_quality`` — sum to 1.0 by construction.
# Ordering matches PRD2 §8 prose: size dominates, freshness and confluence
# matter, kill-zone is a bonus multiplier on the raw score.
FVG_WEIGHT_SIZE = 0.40       # gap size vs ATR
FVG_WEIGHT_FRESHNESS = 0.25  # age in bars (younger = better)
FVG_WEIGHT_CONFLUENCE = 0.25  # nearby OB(s) in same direction
FVG_WEIGHT_RETEST = 0.10     # retest_confirmed flag

# Aging horizon — beyond this many bars the freshness sub-score is 0.
FVG_AGE_HORIZON_BARS = 40

# OB confluence distance (price units). An OB whose midpoint is within
# ``FVG_OB_CONFLUENCE_RANGE × gap_size`` of the FVG midline counts as
# confluent. This is a relative measure so it works for both XAU (0.01 pip)
# and majors (0.0001 pip) without per-instrument config.
FVG_OB_CONFLUENCE_RANGE = 3.0

# Kill-zone score multiplier (multiplicative bonus, capped at 1.0).
FVG_KILL_ZONE_BONUS = 1.15


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QualityCheckResult:
    """Outcome of running the Phase 8 quality gate against a candidate signal.

    ``passed`` is the bottom-line verdict that the generator should use to
    drop or emit the signal. ``reasons`` lists each individual failing check
    in fail-order, so logs and (future) Slack output can show *why* a signal
    was filtered. ``fvg_quality`` / ``fib_pct`` / ``next_tp`` are also surfaced
    so callers can attach them to the SignalPacket for downstream audit even
    when the gate passed.
    """

    passed: bool = True
    fvg_quality: float = 0.0
    in_ote: bool = False
    fib_pct: float = 0.0
    candle_confirmed: bool = False
    next_tp: Optional[float] = None
    reasons: list[str] = field(default_factory=list)
    # Stage 1 always sets this so audits know whether the gate even ran.
    enabled: bool = False


# ---------------------------------------------------------------------------
# Env helpers (mirror the ones in generator.py — kept local to avoid a
# circular import and to let the module stand alone).
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: str) -> bool:
    raw = os.environ.get(name, default).strip()
    return raw not in ("", "0", "false", "False")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# OTE zone check
# ---------------------------------------------------------------------------

def check_ote_zone(
    price: float,
    swing_high: float,
    swing_low: float,
    direction: str,
) -> Tuple[bool, float]:
    """Return ``(in_zone, fib_pct)`` for the Optimal Trade Entry check.

    OTE in ICT is the 0.62-0.79 fibonacci retracement of the last expansion
    leg. For a ``LONG``, the leg is ``swing_low → swing_high`` and the
    retracement is measured from the high back down towards the low. For a
    ``SHORT`` it's the mirror: ``swing_high → swing_low`` with retracement
    measured from the low back up.

    Args:
        price: Current price (entry candidate).
        swing_high: Most recent swing high in the leg.
        swing_low: Most recent swing low in the leg.
        direction: ``"LONG"`` or ``"SHORT"``.

    Returns:
        Tuple ``(in_zone, fib_pct)`` where ``fib_pct`` is the retracement
        percentage in [0.0, 1.0] (or 0.0 on degenerate / invalid input) and
        ``in_zone`` is True when ``0.62 <= fib_pct <= 0.79``.
    """
    direction = (direction or "").upper()
    if direction not in ("LONG", "SHORT"):
        return False, 0.0

    leg = swing_high - swing_low
    if leg <= 0:
        # Degenerate or inverted leg — can't compute a retracement.
        return False, 0.0

    if direction == "LONG":
        # Pullback from the high: fib_pct=0 at the high, 1 at the low.
        fib_pct = (swing_high - price) / leg
    else:  # SHORT
        # Pullback from the low: fib_pct=0 at the low, 1 at the high.
        fib_pct = (price - swing_low) / leg

    # Clamp to a sane range for reporting; values <0 or >1 just mean "outside
    # the leg" which is by definition outside the OTE zone.
    fib_pct = float(fib_pct)
    in_zone = OTE_LOWER_FIB <= fib_pct <= OTE_UPPER_FIB
    return in_zone, fib_pct


# ---------------------------------------------------------------------------
# FVG quality scoring
# ---------------------------------------------------------------------------

def _score_size(gap_size: float, atr: float) -> float:
    """Sub-score: gap size relative to ATR. Maxes out at 1.5× ATR."""
    if atr <= 0:
        return 0.0
    ratio = gap_size / atr
    # Linear ramp from 0 at 0× ATR to 1.0 at 1.5× ATR, then capped.
    return float(min(1.0, max(0.0, ratio / 1.5)))


def _score_freshness(age_bars: int) -> float:
    """Sub-score: freshness. 1.0 at age 0, linearly to 0.0 at horizon."""
    if age_bars <= 0:
        return 1.0
    if age_bars >= FVG_AGE_HORIZON_BARS:
        return 0.0
    return float(1.0 - (age_bars / FVG_AGE_HORIZON_BARS))


def _score_confluence(
    fvg_midline: float,
    fvg_direction: str,
    gap_size: float,
    ob_list: Optional[Sequence] = None,
) -> float:
    """Sub-score: OB confluence. 1.0 if at least one same-direction OB sits
    inside ``FVG_OB_CONFLUENCE_RANGE × gap_size`` of the FVG midline. We don't
    try to count OBs — one confluent block is enough to count as confluence
    in ICT practice, and counting tends to over-reward noisy detection."""
    if not ob_list:
        return 0.0
    if gap_size <= 0:
        return 0.0

    distance_budget = FVG_OB_CONFLUENCE_RANGE * gap_size
    for ob in ob_list:
        ob_dir = getattr(ob, "effective_direction", None) or getattr(ob, "direction", None)
        ob_mid = getattr(ob, "midpoint", None)
        if ob_dir is None or ob_mid is None:
            continue
        if str(ob_dir).upper() != str(fvg_direction).upper():
            continue
        if abs(float(ob_mid) - float(fvg_midline)) <= distance_budget:
            return 1.0
    return 0.0


def score_fvg_quality(
    fvg,
    ob_list: Optional[Sequence] = None,
    atr: float = 0.0,
    kill_zone: bool = False,
    age_bars: Optional[int] = None,
) -> float:
    """Compute a 0.0-1.0 quality score for an FVG zone.

    Multi-factor: size vs ATR, freshness, OB confluence, retest confirmation,
    plus a kill-zone multiplicative bonus (capped at 1.0).

    Args:
        fvg: Object with ``top``, ``bottom``, ``midline``, ``direction``,
            ``age_bars`` (used if ``age_bars`` arg is None), ``retest_confirmed``
            attributes. Designed to accept ``FVGZone`` from ``prism.signal.fvg``
            but anything duck-typed works.
        ob_list: Iterable of ``OrderBlock``-shaped objects for confluence
            scoring. May be empty / None.
        atr: ATR in the same price units as the FVG. Required for size
            scoring; a zero or negative ATR gives a 0.0 size sub-score.
        kill_zone: Whether the candidate signal is firing inside a London/NY
            kill zone. Adds a small multiplicative bonus.
        age_bars: Override ``fvg.age_bars`` (useful for ML feature pipelines
            that don't carry the live age on the dataclass).

    Returns:
        Quality score in [0.0, 1.0].
    """
    try:
        top = float(getattr(fvg, "top"))
        bottom = float(getattr(fvg, "bottom"))
        midline = float(getattr(fvg, "midline", (top + bottom) / 2.0))
        direction = str(getattr(fvg, "direction", "")).upper()
        if age_bars is None:
            age_bars = int(getattr(fvg, "age_bars", 0) or 0)
        retest_confirmed = bool(getattr(fvg, "retest_confirmed", False))
    except (TypeError, ValueError, AttributeError) as exc:
        logger.warning(f"score_fvg_quality: malformed fvg object — {exc}")
        return 0.0

    gap_size = abs(top - bottom)
    size_sub = _score_size(gap_size, float(atr))
    fresh_sub = _score_freshness(int(age_bars))
    conf_sub = _score_confluence(midline, direction, gap_size, ob_list)
    retest_sub = 1.0 if retest_confirmed else 0.0

    base = (
        FVG_WEIGHT_SIZE * size_sub
        + FVG_WEIGHT_FRESHNESS * fresh_sub
        + FVG_WEIGHT_CONFLUENCE * conf_sub
        + FVG_WEIGHT_RETEST * retest_sub
    )

    if kill_zone:
        base *= FVG_KILL_ZONE_BONUS

    # Cap at 1.0 so the bonus never pushes us above the contract.
    return float(min(1.0, max(0.0, base)))


# ---------------------------------------------------------------------------
# Candle / EMA confirmation
# ---------------------------------------------------------------------------

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Plain pandas EMA — small wrapper so the helper is unit-testable."""
    if period <= 0 or len(values) == 0:
        return np.array([], dtype=float)
    return pd.Series(values).ewm(span=period, adjust=False).mean().to_numpy()


def _is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    return (
        prev["close"] < prev["open"]   # previous candle bearish
        and curr["close"] > curr["open"]  # current candle bullish
        and curr["close"] >= prev["open"]
        and curr["open"] <= prev["close"]
    )


def _is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    return (
        prev["close"] > prev["open"]   # previous candle bullish
        and curr["close"] < curr["open"]  # current candle bearish
        and curr["close"] <= prev["open"]
        and curr["open"] >= prev["close"]
    )


def _is_hammer(curr: pd.Series) -> bool:
    """Bullish hammer: small body near top, long lower wick (>= 2× body)."""
    body = abs(curr["close"] - curr["open"])
    if body <= 0:
        return False
    upper_wick = curr["high"] - max(curr["close"], curr["open"])
    lower_wick = min(curr["close"], curr["open"]) - curr["low"]
    return lower_wick >= 2.0 * body and upper_wick <= body


def _is_inverted_hammer(curr: pd.Series) -> bool:
    """Bearish reversal: small body near bottom, long upper wick (>= 2× body)."""
    body = abs(curr["close"] - curr["open"])
    if body <= 0:
        return False
    upper_wick = curr["high"] - max(curr["close"], curr["open"])
    lower_wick = min(curr["close"], curr["open"]) - curr["low"]
    return upper_wick >= 2.0 * body and lower_wick <= body


def check_candle_confirmation(
    df: pd.DataFrame,
    direction: str,
    ema_period: int = 8,
) -> bool:
    """Return True when the last candle shows a reversal pattern AND closes
    on the correct side of the ``ema_period`` EMA for ``direction``.

    "Two confirmations" rule from PRD2 §8 / finastictrading: the candle
    pattern alone never enters; the EMA cross is the second leg.

    Accepted reversal patterns (Stage 1 — keep this conservative; the
    full doji-at-structure / star patterns from PRD2 §8 are Stage 2):

    * LONG: bullish engulfing OR hammer
    * SHORT: bearish engulfing OR inverted hammer

    Args:
        df: OHLC dataframe (must include 'open','high','low','close'). Needs
            at least ``ema_period + 1`` rows for a reliable EMA + a prior bar
            for engulfing comparison.
        direction: ``"LONG"`` or ``"SHORT"``.
        ema_period: EMA period for the close-side confirmation (default 8).

    Returns:
        True iff both reversal-candle and EMA confirmations pass.
    """
    direction = (direction or "").upper()
    if direction not in ("LONG", "SHORT"):
        return False
    if df is None or len(df) < 2:
        return False
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return False

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    if direction == "LONG":
        candle_ok = _is_bullish_engulfing(prev, curr) or _is_hammer(curr)
    else:
        candle_ok = _is_bearish_engulfing(prev, curr) or _is_inverted_hammer(curr)
    if not candle_ok:
        return False

    ema_vals = _ema(df["close"].to_numpy(), ema_period)
    if len(ema_vals) == 0:
        return False
    last_close = float(curr["close"])
    last_ema = float(ema_vals[-1])
    if direction == "LONG":
        return last_close > last_ema
    return last_close < last_ema


# ---------------------------------------------------------------------------
# Dynamic liquidity-pool target
# ---------------------------------------------------------------------------

def _coerce_swing_level(swing) -> Optional[float]:
    """Accept either a float, a dict with ``price``/``level``/``value``, or
    an object with the same attributes. Returns None on unparseable input."""
    if swing is None:
        return None
    if isinstance(swing, (int, float, np.floating, np.integer)):
        return float(swing)
    if isinstance(swing, dict):
        for k in ("price", "level", "value", "high", "low"):
            if k in swing and swing[k] is not None:
                try:
                    return float(swing[k])
                except (TypeError, ValueError):
                    continue
        return None
    for attr in ("price", "level", "value", "high", "low"):
        v = getattr(swing, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def next_liquidity_pool(
    swing_points: Iterable,
    direction: str,
    current_price: float,
) -> Optional[float]:
    """Return the nearest unconsumed liquidity pool above (for LONG) or below
    (for SHORT) ``current_price``.

    "Liquidity pool" = an unbroken swing high (LONG target) or swing low
    (SHORT target). The caller is responsible for filtering out already-
    consumed pools before passing them in — this helper just picks the
    nearest remaining one in the trade direction. That keeps the function
    pure and trivially testable.

    Args:
        swing_points: Iterable of swing levels. Each element can be a float,
            a dict with ``price``/``level``/``high``/``low`` keys, or an
            object with the same attributes.
        direction: ``"LONG"`` or ``"SHORT"``.
        current_price: Current entry price.

    Returns:
        Nearest swing level beyond ``current_price`` in the trade direction,
        or ``None`` if no qualifying swing exists.
    """
    direction = (direction or "").upper()
    if direction not in ("LONG", "SHORT") or swing_points is None:
        return None

    candidates: list[float] = []
    for sp in swing_points:
        lvl = _coerce_swing_level(sp)
        if lvl is None:
            continue
        if direction == "LONG" and lvl > current_price:
            candidates.append(lvl)
        elif direction == "SHORT" and lvl < current_price:
            candidates.append(lvl)

    if not candidates:
        return None
    if direction == "LONG":
        return float(min(candidates))  # closest above
    return float(max(candidates))      # closest below


# ---------------------------------------------------------------------------
# Stage 1 orchestrator
# ---------------------------------------------------------------------------

def apply_quality_filter(
    *,
    fvg_zone,
    direction: str,
    entry_price: float,
    swing_high: float,
    swing_low: float,
    entry_df: pd.DataFrame,
    ob_list: Optional[Sequence] = None,
    atr: float = 0.0,
    kill_zone: bool = False,
    swing_points: Optional[Iterable] = None,
) -> QualityCheckResult:
    """Stage 1 entry point used by ``SignalGenerator``.

    Reads the master switch ``PRISM_QUALITY_FILTER_ENABLED`` (default ``"0"``)
    and short-circuits to a pass-through result when off — the generator can
    still pull values like ``fvg_quality`` off the result for observability
    without changing behavior.

    When enabled, runs the four PRD2 §8 checks and assembles the result:

    1. FVG quality score ≥ ``PRISM_FVG_MIN_QUALITY`` (default 0.5).
    2. OTE zone — preferred but not hard-required if
       ``PRISM_OTE_REQUIRED=1`` (default ``0``).
    3. Reversal candle + 8 EMA confirmation, gated by
       ``PRISM_REQUIRE_CANDLE_CONFIRM`` (default ``1``).
    4. Dynamic liquidity-pool TP — informational only in Stage 1, populated
       on the result so the generator can hand it to the SignalPacket.

    Fail-soft: any sub-check raising is logged at ``warning`` and recorded
    as a fail reason; the gate still returns a structured result.
    """
    enabled = _env_bool("PRISM_QUALITY_FILTER_ENABLED", "0")
    result = QualityCheckResult(enabled=enabled)

    # Compute the informational fields whether enabled or not — cheap, and
    # downstream observability (PRD2 Phase 7.B features) wants them either
    # way. The *gating* is what's gated on ``enabled``.
    try:
        result.fvg_quality = score_fvg_quality(
            fvg_zone, ob_list=ob_list, atr=atr, kill_zone=kill_zone
        )
    except Exception as exc:
        logger.warning(f"score_fvg_quality raised: {exc}")
        result.fvg_quality = 0.0

    try:
        in_ote, fib_pct = check_ote_zone(
            entry_price, swing_high, swing_low, direction
        )
        result.in_ote = in_ote
        result.fib_pct = fib_pct
    except Exception as exc:
        logger.warning(f"check_ote_zone raised: {exc}")

    try:
        result.candle_confirmed = check_candle_confirmation(entry_df, direction)
    except Exception as exc:
        logger.warning(f"check_candle_confirmation raised: {exc}")

    try:
        result.next_tp = next_liquidity_pool(
            swing_points or [], direction, entry_price
        )
    except Exception as exc:
        logger.warning(f"next_liquidity_pool raised: {exc}")

    if not enabled:
        # Stage 1, flag off: pass-through. Generator behavior unchanged.
        result.passed = True
        return result

    # ---- Stage 1, flag on: actually gate. ------------------------------
    min_quality = _env_float("PRISM_FVG_MIN_QUALITY", DEFAULT_MIN_FVG_QUALITY)
    require_ote = _env_bool("PRISM_OTE_REQUIRED", "0")
    require_candle = _env_bool("PRISM_REQUIRE_CANDLE_CONFIRM", "1")

    reasons: list[str] = []
    if result.fvg_quality < min_quality:
        reasons.append(
            f"fvg_quality {result.fvg_quality:.2f} < min {min_quality:.2f}"
        )
    if require_ote and not result.in_ote:
        reasons.append(f"entry outside OTE zone (fib {result.fib_pct:.2f})")
    if require_candle and not result.candle_confirmed:
        reasons.append("no candle/EMA confirmation")

    result.reasons = reasons
    result.passed = not reasons
    return result
