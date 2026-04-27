"""Per-signal structured audit log (Phase 6.F).

Persists every signal that the runner actually fires to a daily JSONL file
under ``PRISM_STATE_DIR/signal_audit/<instrument>/YYYY-MM-DD.jsonl``.

Why this exists
---------------
Phase 7.A §4.3 makes this the prerequisite for the validation data path:
gate 5 of the retrain acceptance compares the live distribution recorded
here against the historical-replay reconstruction (see PHASE_7A_SCOPE.md
§6.1). Without a structured live log, "live vs. historical drift" is
unobservable and Phase 7.A cannot promote a new model.

Population semantics
--------------------
The runner calls :func:`write_signal_audit` *after* the duplicate-suppression
guard (``runner._should_fire``), so each audit row corresponds to a
**signal-producing bar** — exactly the population PHASE_7A_SCOPE.md §6.1
commits to as the live side of the drift comparison. The historical
replay-mode builder must condition on the same selection criteria so the
two distributions are comparable.

Failure semantics
-----------------
A broken audit logger MUST NOT take down signal delivery. Disk full,
read-only mount, JSON serialization failure on a stray numpy/pandas type —
all are caught, logged at ``ERROR`` with a traceback, and swallowed. The
runner continues to send the signal to Slack / MT5 as normal.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prism.execution.mt5_bridge import SignalPacket

logger = logging.getLogger(__name__)


# Subset of ``SignalPacket`` written to each audit record. Whitelisted
# explicitly (rather than ``asdict`` + filter) so accidental additions to
# ``SignalPacket`` don't silently leak unrelated fields into the audit
# JSONL — the schema is part of the Phase 7.A contract and changes here
# need to be paired with changes to the historical-replay builder.
AUDIT_FIELDS = (
    "instrument",
    "direction",
    "confidence",
    "confidence_level",
    "signal_id",
    "signal_time",
    "model_version",
    "regime",
    "news_bias",
    "htf_bias",
    "smart_money",
)


def _enabled() -> bool:
    """``PRISM_SIGNAL_AUDIT_ENABLED`` defaults ON. Phase 7.A treats this
    as a Stage-1-rollout prerequisite, so the safe default is to write."""
    raw = os.environ.get("PRISM_SIGNAL_AUDIT_ENABLED", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _state_dir() -> Path:
    return Path(os.environ.get("PRISM_STATE_DIR", "state"))


def audit_path(instrument: str, when: Optional[datetime] = None) -> Path:
    """Resolve the JSONL file for ``instrument`` on ``when`` (UTC).

    Daily rotation is keyed off the audit timestamp's UTC date, *not* the
    signal's bar timestamp — a signal generated on the H4 bar at
    23:55 UTC and audited at 00:01 UTC the next day lands in the new
    day's file. That matches operator intuition ("today's audit log") and
    keeps the file boundary aligned with log-shipping cron jobs.
    """
    when = when or datetime.now(timezone.utc)
    day = when.strftime("%Y-%m-%d")
    return _state_dir() / "signal_audit" / instrument / f"{day}.jsonl"


def write_signal_audit(
    signal: "SignalPacket",
    *,
    when: Optional[datetime] = None,
) -> Optional[Path]:
    """Append ``signal`` to the audit JSONL for its instrument.

    Returns the file path on success, ``None`` when audit is disabled or
    the write fails. Never raises — the runner relies on this contract.
    """
    if not _enabled():
        return None

    try:
        when = when or datetime.now(timezone.utc)
        path = audit_path(signal.instrument, when=when)
        record = {"audit_ts": when.isoformat()}
        for field in AUDIT_FIELDS:
            record[field] = getattr(signal, field, None)

        path.parent.mkdir(parents=True, exist_ok=True)
        # ``default=str`` coerces stray non-JSON-serializable values
        # (numpy scalars, pandas Timestamps, enums) so a single bad
        # field doesn't burn the whole record. POSIX append mode +
        # one-line records keeps writes atomic up to PIPE_BUF (~4KB),
        # which is well above any realistic audit-line size.
        line = json.dumps(record, default=str, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return path
    except Exception as exc:  # noqa: BLE001 — audit must not propagate
        logger.error(
            "signal_audit: failed to persist %s/%s signal: %s",
            getattr(signal, "instrument", "?"),
            getattr(signal, "signal_id", "?"),
            exc,
            exc_info=True,
        )
        return None
