"""
PRISM Daily Drawdown Guard

Halts new signal generation once the realized loss for the current UTC
trading day breaches a configured threshold. The philosophy is simple:
a bad news day should cost at most N% of the starting balance, not the
account. The guard does NOT close existing positions — closing a losing
trade early is a separate risk decision that depends on the strategy's
edge expectancy. The guard only stops opening new ones.

Trip conditions (whichever fires first):
  * ``realized_pnl_usd <= -(start_of_day_balance * max_daily_loss_pct)``
    e.g. -3% of a $10,000 balance = halt after -$300 realized.
  * ``realized_pnl_usd <= -max_daily_loss_usd`` (absolute USD cap, optional)

State is persisted to ``{PRISM_STATE_DIR}/daily_drawdown.json`` so a
mid-day runner restart doesn't reset the counter and re-enable trading
after a halt. On a new UTC day the counter resets automatically.

Realized PnL sources:
  * **Live MT5**: ``bridge.deals_since_utc_midnight()`` — canonical,
    picks up SL/TP hits that happen async on the broker side.
  * **Mock / dev**: manually-recorded PnL via ``record_manual(pnl_usd)``.
    Lets tests exercise trip conditions without a live terminal.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class _GuardState:
    """Snapshot of what the guard believes about today."""
    date: str                           # UTC ISO date: "2026-04-21"
    start_of_day_balance: float         # Balance at first observation today
    realized_pnl_usd: float = 0.0       # Cumulative for the day
    trip_at: Optional[str] = None       # ISO ts when first tripped (or None)
    notified: bool = False              # Slack trip notification already sent?
    known_tickets: list = field(default_factory=list)  # Deals already counted


class DrawdownGuard:
    """
    Daily drawdown circuit breaker. Instantiate once per runner; call
    ``refresh(now)`` and check ``is_tripped`` at the top of each scan.

    Args:
        bridge: An MT5Bridge (or MockMT5Bridge). Must expose
            ``get_account_balance()`` and ``deals_since_utc_midnight()``.
        state_dir: Path to persist guard state. Created if absent.
        max_daily_loss_pct: Halt after realized loss ≥ pct of SOD balance.
            Default 0.03 = 3%. Effectively ≈3R since each trade is sized
            to 1% risk.
        max_daily_loss_usd: Optional absolute USD cap (positive number).
            Used as an OR condition. Set to None (default) to disable.
        magic_number: PRISM's magic number, passed through to the bridge
            so hand-placed trades don't count against the kill-switch.
    """

    STATE_FILENAME = "daily_drawdown.json"

    def __init__(
        self,
        bridge,
        state_dir,
        max_daily_loss_pct: float = 0.03,
        max_daily_loss_usd: Optional[float] = None,
        magic_number: Optional[int] = None,
    ):
        self.bridge = bridge
        self.state_dir = Path(state_dir)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_daily_loss_usd = (
            float(max_daily_loss_usd) if max_daily_loss_usd else None
        )
        # Default comes from the bridge module; avoid a hard import so tests
        # can run this file in isolation.
        if magic_number is None:
            from prism.execution.mt5_bridge import MAGIC_NUMBER
            magic_number = MAGIC_NUMBER
        self.magic_number = int(magic_number)

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state: Optional[_GuardState] = self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def _state_path(self) -> Path:
        return self.state_dir / self.STATE_FILENAME

    def _load_state(self) -> Optional[_GuardState]:
        path = self._state_path
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            return _GuardState(**payload)
        except Exception as e:
            logger.warning("Corrupt drawdown state at %s: %s — starting fresh", path, e)
            return None

    def _persist(self) -> None:
        if self._state is None:
            return
        try:
            self._state_path.write_text(json.dumps(asdict(self._state), indent=2))
        except Exception as e:
            logger.warning("Failed to persist drawdown state: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, now: Optional[datetime] = None) -> None:
        """
        Top-of-scan refresh. Resets state if we've crossed into a new
        UTC day. Pulls live deal history from MT5 when available. Safe
        to call on every scan (idempotent for same-day calls).
        """
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("refresh requires tz-aware datetime")
        today = now.astimezone(timezone.utc).date().isoformat()

        if self._state is None or self._state.date != today:
            # First observation of the day — snapshot balance so we know
            # the denominator for the pct loss threshold.
            try:
                balance = float(self.bridge.get_account_balance() or 0.0)
            except Exception as e:
                logger.warning("Guard: get_account_balance failed at reset: %s", e)
                balance = 0.0
            self._state = _GuardState(
                date=today,
                start_of_day_balance=balance,
                realized_pnl_usd=0.0,
                trip_at=None,
                notified=False,
                known_tickets=[],
            )
            self._persist()
            logger.info(
                "Drawdown guard: new day %s, SOD balance=$%.2f, threshold=%.1f%%",
                today, balance, self.max_daily_loss_pct * 100,
            )
            # Warn when SOD snapshot is taken mid-day (runner started
            # after midnight). The threshold is relative to this balance,
            # NOT the real open-of-day balance, which may mislead review.
            if now.hour > 0 or now.minute > 5:
                logger.info(
                    "Drawdown guard: SOD balance snapshot taken at %s UTC "
                    "(mid-day start). The -%d%% threshold is relative to "
                    "this balance, not the actual open-of-day balance.",
                    now.strftime("%H:%M"), int(self.max_daily_loss_pct * 100),
                )

        # Live-bridge deal sync. Mock bridge returns [] so this is a no-op
        # for unit tests unless they monkeypatch deals_since_utc_midnight.
        try:
            deals = self.bridge.deals_since_utc_midnight(
                now=now, magic_number=self.magic_number,
            ) or []
        except Exception as e:
            logger.warning("Guard: deals_since_utc_midnight failed: %s", e)
            deals = []

        if deals:
            new_tickets = set()
            added_pnl = 0.0
            for d in deals:
                ticket = d.get("ticket")
                if ticket is None or ticket in self._state.known_tickets:
                    continue
                added_pnl += float(d.get("profit", 0.0))
                new_tickets.add(ticket)
            if new_tickets:
                self._state.realized_pnl_usd += added_pnl
                self._state.known_tickets.extend(sorted(new_tickets))
                logger.info(
                    "Guard: recorded %d new deals, +$%.2f, cumulative=$%.2f",
                    len(new_tickets), added_pnl, self._state.realized_pnl_usd,
                )
                self._check_trip(now)
                self._persist()

    def record_manual(self, pnl_usd: float, now: Optional[datetime] = None) -> None:
        """
        Record a trade outcome manually. Used by Mock bridge paths and
        tests where there's no deal history to sync. Not needed in
        production — ``refresh()`` is the canonical source.
        """
        now = now or datetime.now(timezone.utc)
        self.refresh(now)  # Ensure state exists + current day
        assert self._state is not None
        self._state.realized_pnl_usd += float(pnl_usd)
        self._check_trip(now)
        self._persist()

    def mark_notified(self) -> None:
        """Callers invoke this after posting the Slack trip alert."""
        if self._state is not None:
            self._state.notified = True
            self._persist()

    @property
    def is_tripped(self) -> bool:
        if self._state is None:
            return False
        return self._state.trip_at is not None

    @property
    def needs_notification(self) -> bool:
        """Tripped AND we haven't notified yet — one-shot semantics."""
        return self.is_tripped and not (self._state and self._state.notified)

    @property
    def snapshot(self) -> dict:
        """Serializable view of the guard's current state."""
        if self._state is None:
            return {
                "date": None, "start_of_day_balance": 0.0,
                "realized_pnl_usd": 0.0, "trip_at": None, "notified": False,
            }
        s = self._state
        return {
            "date": s.date,
            "start_of_day_balance": s.start_of_day_balance,
            "realized_pnl_usd": s.realized_pnl_usd,
            "trip_at": s.trip_at,
            "notified": s.notified,
            "trip_threshold_usd": self._trip_threshold_usd(),
        }

    def format_alert(self) -> str:
        """Human-readable Slack alert text when the guard trips."""
        s = self._state
        if s is None:
            return ":octagonal_sign: Drawdown guard tripped (no state)"
        loss_pct = (
            abs(s.realized_pnl_usd) / s.start_of_day_balance * 100
            if s.start_of_day_balance > 0 else 0.0
        )
        return (
            f":octagonal_sign: *PRISM halted for today* — "
            f"daily drawdown limit hit.\n"
            f"Realized PnL: ${s.realized_pnl_usd:+.2f} "
            f"({loss_pct:.1f}% of ${s.start_of_day_balance:,.2f} SOD)\n"
            f"Threshold: {self.max_daily_loss_pct * 100:.1f}%"
            + (f" or ${self.max_daily_loss_usd:,.2f} absolute" if self.max_daily_loss_usd else "")
            + f"\nTripped at {s.trip_at} UTC. Will reset at next UTC midnight."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _trip_threshold_usd(self) -> float:
        """
        Return the current effective trip threshold in USD as a negative
        number (e.g. ``-300.0`` means "halt when realized_pnl <= -$300").

        When both pct and absolute caps are set, the tighter one fires
        first. "Trip if realized <= pct_cap OR realized <= abs_cap" is
        algebraically equivalent to "trip if realized <= max(pct_cap,
        abs_cap)" — the LESS negative number fires sooner, so ``max``
        (not ``min``) is correct.
        """
        if self._state is None:
            return 0.0
        pct_cap = -self._state.start_of_day_balance * self.max_daily_loss_pct
        if self.max_daily_loss_usd is None:
            return pct_cap
        abs_cap = -self.max_daily_loss_usd
        return max(pct_cap, abs_cap)

    def _check_trip(self, now: datetime) -> None:
        if self._state is None or self._state.trip_at is not None:
            return
        if self._state.realized_pnl_usd <= self._trip_threshold_usd():
            self._state.trip_at = now.astimezone(timezone.utc).isoformat()
            logger.error(
                "DRAWDOWN GUARD TRIPPED: realized=$%.2f threshold=$%.2f — "
                "halting new entries until UTC midnight",
                self._state.realized_pnl_usd, self._trip_threshold_usd(),
            )


# ---------------------------------------------------------------------------
# Factory helper used by the runner — reads env for thresholds + state dir.
# Kept out of the class so tests can construct the guard directly without
# touching os.environ.
# ---------------------------------------------------------------------------

def build_guard_from_env(bridge, state_dir: Path) -> DrawdownGuard:
    pct = float(os.environ.get("PRISM_MAX_DAILY_LOSS_PCT", "0.03"))
    usd_raw = os.environ.get("PRISM_MAX_DAILY_LOSS_USD", "").strip()
    usd = float(usd_raw) if usd_raw else None
    return DrawdownGuard(
        bridge=bridge,
        state_dir=state_dir,
        max_daily_loss_pct=pct,
        max_daily_loss_usd=usd,
    )
