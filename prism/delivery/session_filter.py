"""
PRISM Session Filter
Only allows trades during high-liquidity kill zones (per @tradesbysci ICC rules).

Kill zones:
- London: 07:00 – 11:00 UTC
- New York: 13:00 – 17:00 UTC
- Asian: SKIP (low liquidity, avoid)

London and NY kill zones do not overlap (gap 11:00–13:00 UTC).

Sunday-open gap:
  FX re-opens around Sunday 22:00 UTC after the weekend. The first
  ~30 minutes of the week carry wide spreads, low liquidity, and
  weekend-gap prints — terrible execution quality. ``is_sunday_open_gap``
  flags this window so the runner can skip it explicitly, independent
  of kill-zone logic (a safety net for future modes that might scan
  outside London/NY).
"""
import os
from datetime import datetime, time, timedelta, timezone
from enum import Enum


class Session(str, Enum):
    LONDON = "London"
    NEW_YORK = "New York"
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
    """
    Return the active trading session for a given UTC datetime.

    Parameters
    ----------
    dt : datetime, optional
        A **timezone-aware** datetime. If omitted, ``datetime.now(timezone.utc)``
        is used. Passing a naive datetime raises ``ValueError`` — always pass
        tz-aware values (e.g. ``datetime.now(timezone.utc)`` or any
        ``pytz``/``zoneinfo`` aware object).
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        raise ValueError(
            "get_current_session requires a tz-aware datetime (use timezone.utc)"
        )
    # Normalise to UTC before extracting wall-clock time
    t = dt.astimezone(timezone.utc).time().replace(tzinfo=None)

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
    return session in (Session.LONDON, Session.NEW_YORK)


def session_label(dt: datetime = None) -> str:
    """Human-readable session label for signal output."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    session = get_current_session(dt)
    time_str = dt.astimezone(timezone.utc).strftime("%H:%M") + " UTC"
    if session == Session.OFF:
        return f"Off-session ({time_str})"
    return f"{session.value} Kill Zone ({time_str})"


# ---------------------------------------------------------------------------
# Sunday-open gap guard
# ---------------------------------------------------------------------------

# Canonical FX re-open on the Exness/retail side. DST drifts the real
# Sydney open by an hour, but broker quoting resumes on this UTC slot
# year-round, so a fixed 22:00 UTC reference is correct for PRISM.
SUN_OPEN_UTC_HOUR = 22
_DEFAULT_SUN_OPEN_SKIP_MIN = 30


def _resolve_skip_minutes(skip_minutes):
    if skip_minutes is not None:
        return int(skip_minutes)
    raw = os.environ.get("PRISM_SUN_OPEN_SKIP_MIN", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return _DEFAULT_SUN_OPEN_SKIP_MIN


def is_sunday_open_gap(dt: datetime = None, skip_minutes: int = None) -> bool:
    """
    True when ``dt`` falls inside the Sunday-open gap window — the first
    ``skip_minutes`` after ``22:00 UTC`` on Sunday. During this window
    PRISM skips scans entirely:

    * Brokers widen spreads aggressively to manage rollover risk, so
      even profitable signals get eaten by slippage.
    * Tick volume prints on low liquidity — bar data is unreliable and
      any freshness check can falsely pass on stale quotes.
    * Weekend-gap moves between Friday close and Sunday open distort
      ATR, EMAs, and the FVG detector for the first few bars.

    Parameters
    ----------
    dt : datetime, optional
        A timezone-aware datetime. Defaults to ``datetime.now(timezone.utc)``.
        A naive datetime raises ``ValueError`` — consistent with the rest
        of this module.
    skip_minutes : int, optional
        Override the window length. When omitted, reads
        ``PRISM_SUN_OPEN_SKIP_MIN`` (default 30).

    Notes
    -----
    Returns False on every non-Sunday and every Sunday time outside the
    window — callers can treat this as a pure gate. The current PRISM
    kill zones (London + NY) don't overlap with Sunday 22:00 UTC, so
    this is defense-in-depth today; if kill zones are ever expanded to
    include Sydney/Tokyo, this guard prevents running against the open.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        raise ValueError(
            "is_sunday_open_gap requires a tz-aware datetime (use timezone.utc)"
        )
    dt_utc = dt.astimezone(timezone.utc)
    # weekday(): Monday=0, Sunday=6.
    if dt_utc.weekday() != 6:
        return False
    skip = _resolve_skip_minutes(skip_minutes)
    if skip <= 0:
        return False
    open_start = dt_utc.replace(
        hour=SUN_OPEN_UTC_HOUR, minute=0, second=0, microsecond=0,
    )
    open_end = open_start + timedelta(minutes=skip)
    return open_start <= dt_utc < open_end
