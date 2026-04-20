"""
PRISM Session Filter
Only allows trades during high-liquidity kill zones (per @tradesbysci ICC rules).

Kill zones:
- London: 07:00 – 11:00 UTC
- New York: 13:00 – 17:00 UTC
- Asian: SKIP (low liquidity, avoid)
- London/NY Overlap: 13:00 – 15:00 UTC (highest probability)
"""
from datetime import datetime, time, timezone
from enum import Enum


class Session(str, Enum):
    LONDON = "London"
    NEW_YORK = "New York"
    LONDON_NY_OVERLAP = "London/NY Overlap"
    ASIAN = "Asian"
    OFF = "Off-session"


# Kill zone windows (UTC)
LONDON_START = time(7, 0)
LONDON_END = time(11, 0)
NY_START = time(13, 0)
NY_END = time(17, 0)
ASIAN_START = time(0, 0)
ASIAN_END = time(6, 0)


def get_current_session(dt: datetime = None) -> Session:
    """Return the active trading session for a given UTC datetime."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    t = dt.time().replace(tzinfo=None)

    # London/NY overlap: 13:00-15:00 UTC (both London and NY open)
    if NY_START <= t < time(15, 0) and LONDON_START <= t < LONDON_END:
        return Session.LONDON_NY_OVERLAP

    # New York: 13:00-17:00 UTC
    if NY_START <= t < NY_END:
        return Session.NEW_YORK

    # London: 07:00-11:00 UTC
    if LONDON_START <= t < LONDON_END:
        return Session.LONDON

    # Asian: 00:00-06:00 UTC
    if ASIAN_START <= t < ASIAN_END:
        return Session.ASIAN

    return Session.OFF


def is_kill_zone(dt: datetime = None) -> bool:
    """
    Returns True only if current time is in London or NY kill zone.
    PRISM only fires signals during kill zones (per ICC rules).
    """
    session = get_current_session(dt)
    return session in (Session.LONDON, Session.NEW_YORK, Session.LONDON_NY_OVERLAP)


def session_label(dt: datetime = None) -> str:
    """Human-readable session label for signal output."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    session = get_current_session(dt)
    time_str = dt.strftime("%H:%M") + " UTC"
    if session == Session.OFF:
        return f"Off-session ({time_str})"
    return f"{session.value} Kill Zone ({time_str})"
