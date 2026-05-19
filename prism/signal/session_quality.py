"""Session Quality Scorer — the LRF (Liquidity Resistance Factor) equivalent.

Scores the current session's tradeability on a 0–20 point scale using
economic calendar impact, market conditions, and Asian range characteristics.

Scoring tiers (Gatie-inspired):
  0–4   → FAVORABLE   — standard thresholds, full position size
  5–9   → CAUTIOUS    — raise min confidence, reduce size
  10+   → SKIP        — don't trade this session (FOMC, NFP, etc.)

This replaces the binary ``news_blackout`` with a graduated filter.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prism.news.intelligence import NewsSignal
    from prism.signal.session_bias import AsianRange

logger = logging.getLogger(__name__)


class SessionGrade(str, Enum):
    FAVORABLE = "FAVORABLE"
    CAUTIOUS = "CAUTIOUS"
    SKIP = "SKIP"


# Event names that signal maximum-risk sessions (10 pts each).
_CRITICAL_EVENTS = frozenset({
    "fomc", "federal funds rate", "fed interest rate",
    "nonfarm payrolls", "non-farm payrolls", "nfp",
    "cpi", "consumer price index",
    "powell", "fed chair",
})

# Medium-impact events (3 pts each).
_MEDIUM_EVENTS = frozenset({
    "ppi", "producer price index",
    "retail sales",
    "gdp", "gross domestic product",
    "ism manufacturing", "ism services",
    "jolts", "job openings",
    "initial jobless claims", "unemployment claims",
    "ecb", "boe", "boj",
})


@dataclass
class SessionQuality:
    """Result of the session quality scoring."""
    score: int
    grade: SessionGrade
    min_confidence: float
    reasons: list[str] = field(default_factory=list)


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


def score_session(
    news_signal: "NewsSignal",
    asian_range: Optional["AsianRange"] = None,
    prior_session_range_pips: Optional[float] = None,
) -> SessionQuality:
    """Compute the session quality score.

    Parameters
    ----------
    news_signal
        From ``NewsIntelligence.get_signal()``.
    asian_range
        Today's Asian range from ``SessionBiasEngine``.
    prior_session_range_pips
        Range of the prior kill-zone session in pips (if available).
    """
    points = 0
    reasons: list[str] = []

    # --- Economic calendar (from existing news_signal) ---
    event_name_lower = (news_signal.event_name or "").lower()
    if news_signal.event_flag:
        if any(kw in event_name_lower for kw in _CRITICAL_EVENTS):
            points += 10
            reasons.append(f"critical event: {news_signal.event_name}")
        else:
            points += 5
            reasons.append(f"high-impact event: {news_signal.event_name}")

    # --- Geopolitical ---
    if news_signal.geopolitical_active:
        points += 3
        reasons.append("geopolitical risk active")

    # --- Asian range characteristics ---
    wide_threshold = _env_float("PRISM_ASIAN_WIDE_PIPS", 80.0)
    tight_threshold = _env_float("PRISM_ASIAN_TIGHT_PIPS", 15.0)

    if asian_range is not None:
        if asian_range.range_pips > wide_threshold:
            points += 3
            reasons.append(
                f"wide Asian range: {asian_range.range_pips:.0f} pips "
                f"(>{wide_threshold:.0f})"
            )
        elif asian_range.range_pips < tight_threshold:
            # Extremely tight range could mean no real accumulation,
            # but also could mean a compressed spring. Score it neutral
            # but note it.
            reasons.append(
                f"tight Asian range: {asian_range.range_pips:.0f} pips — "
                f"compressed spring or no accumulation"
            )

    # --- Prior session expansion ---
    expansion_threshold = _env_float("PRISM_PRIOR_EXPANSION_PIPS", 150.0)
    if prior_session_range_pips is not None and prior_session_range_pips > expansion_threshold:
        points += 2
        reasons.append(
            f"prior session expanded {prior_session_range_pips:.0f} pips "
            f"(>{expansion_threshold:.0f}) — exhaustion risk"
        )

    # --- Grade ---
    skip_threshold = _env_int("PRISM_LRF_SKIP_THRESHOLD", 10)
    cautious_threshold = _env_int("PRISM_LRF_CAUTIOUS_THRESHOLD", 5)

    if points >= skip_threshold:
        grade = SessionGrade.SKIP
    elif points >= cautious_threshold:
        grade = SessionGrade.CAUTIOUS
    else:
        grade = SessionGrade.FAVORABLE

    # --- Min confidence per grade ---
    base_conf = _env_float("PRISM_MIN_CONFIDENCE", 0.60)
    cautious_conf = _env_float("PRISM_CAUTIOUS_MIN_CONFIDENCE", 0.70)

    if grade == SessionGrade.SKIP:
        min_confidence = 1.0  # unreachable — blocks all
    elif grade == SessionGrade.CAUTIOUS:
        min_confidence = cautious_conf
    else:
        min_confidence = base_conf

    logger.info(
        "Session quality: score=%d grade=%s min_conf=%.2f%s",
        points, grade.value, min_confidence,
        f" ({'; '.join(reasons)})" if reasons else "",
    )

    return SessionQuality(
        score=points,
        grade=grade,
        min_confidence=min_confidence,
        reasons=reasons,
    )
