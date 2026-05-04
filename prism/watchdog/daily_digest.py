"""PRISM daily self-diagnosis digest.

Posts a one-block Slack health report at 08:00 UTC every day. Designed so
Brian and Ada can wake up, glance at Slack, and know whether PRISM is
healthy without ever logging into the VPS. Sun-Tzu rule of thumb:
*supreme excellence is invisible.*

Inputs (all optional, all fail-soft):
  - ``logs/watchdog.log`` — restart events for the trailing 24h
  - ``state/signal_audit/<INSTRUMENT>/YYYY-MM-DD.jsonl`` — fired signals
  - ``models/manifest_<INSTRUMENT>.json`` — last retrain timestamp
  - ``PRISM_EXECUTION_MODE`` env var — current mode (default ``NOTIFY``)

The Slack message follows the format pinned in the action plan:

    📊 *PRISM Daily — 2026-05-05*

    *Runner:* ✅ Up 100% (0 restarts)
    *Signals (24h):* XAUUSD: 3 | EURUSD: 2
    *Confidence:* HIGH: 1 | MEDIUM: 3 | LOW: 1
    *Last retrain:* XAUUSD 2026-05-04 | EURUSD 2026-05-04
    *Mode:* NOTIFY
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("prism.daily_digest")

DEFAULT_LOG_PATH = "logs/daily_digest.log"
DEFAULT_WATCHDOG_LOG = "logs/watchdog.log"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _instruments() -> List[str]:
    raw = os.environ.get("PRISM_INSTRUMENTS", "XAUUSD,EURUSD")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _state_dir() -> Path:
    return Path(os.environ.get("PRISM_STATE_DIR", "state"))


def _models_dir() -> Path:
    return Path(os.environ.get("PRISM_MODELS_DIR", "models"))


def _execution_mode() -> str:
    return os.environ.get("PRISM_EXECUTION_MODE", "NOTIFY")


# ---------------------------------------------------------------------------
# Audit log digest
# ---------------------------------------------------------------------------
def _read_audit_records(
    instrument: str,
    *,
    today: datetime,
    state_dir: Optional[Path] = None,
) -> List[dict]:
    """Read the JSONL for the trailing 24h (today's file)."""
    root = (state_dir or _state_dir()) / "signal_audit" / instrument
    path = root / f"{today.strftime('%Y-%m-%d')}.jsonl"
    if not path.exists():
        return []
    out: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.error("digest: cannot read %s: %s", path, exc)
    return out


def signals_and_confidence(
    instruments: Sequence[str],
    *,
    today: Optional[datetime] = None,
    state_dir: Optional[Path] = None,
) -> Tuple[Dict[str, int], Counter]:
    """Return (per-instrument signal counts, confidence-level counter)."""
    today = today or datetime.now(timezone.utc)
    counts: Dict[str, int] = {}
    conf: Counter = Counter()
    for inst in instruments:
        records = _read_audit_records(inst, today=today, state_dir=state_dir)
        counts[inst] = len(records)
        for r in records:
            level = (r.get("confidence_level") or "").upper()
            if level in ("HIGH", "MEDIUM", "LOW"):
                conf[level] += 1
    return counts, conf


# ---------------------------------------------------------------------------
# Watchdog log digest
# ---------------------------------------------------------------------------
# Matches lines emitted by handle_runner_down() in watchdog.py:
#   "Recovery succeeded on attempt 1/3"
_RECOVERY_LINE = re.compile(r"Recovery succeeded")
# Matches the leading ISO timestamp our FileHandler writes via _configure_logging:
#   2026-05-05T08:00:01+0000 [INFO] prism.watchdog — ...
_LEADING_ISO_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def watchdog_restarts_in_window(
    *,
    log_path: Optional[Path] = None,
    today: Optional[datetime] = None,
    window_hours: int = 24,
) -> int:
    """Count successful runner restarts in the trailing window."""
    path = Path(log_path or os.environ.get("PRISM_WATCHDOG_LOG", DEFAULT_WATCHDOG_LOG))
    if not path.exists():
        return 0
    cutoff = (today or datetime.now(timezone.utc)) - timedelta(hours=window_hours)
    count = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not _RECOVERY_LINE.search(line):
                    continue
                m = _LEADING_ISO_TS.match(line)
                if not m:
                    # No timestamp parsable — count it; better to over-report
                    # restarts than to silently lose them.
                    count += 1
                    continue
                try:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
                    ts = ts.replace(tzinfo=timezone.utc)
                except ValueError:
                    count += 1
                    continue
                if ts >= cutoff:
                    count += 1
    except OSError as exc:
        logger.error("digest: cannot read %s: %s", path, exc)
    return count


def runner_uptime_pct(restarts: int) -> int:
    """Crude uptime: 100% if zero restarts, else heuristic.

    Each successful restart is treated as ~5 minutes of downtime
    (one watchdog cycle + verify wait), capped at 100/0.
    """
    if restarts <= 0:
        return 100
    minutes_down = restarts * 5
    minutes_in_day = 24 * 60
    pct = max(0, 100 - int(round((minutes_down / minutes_in_day) * 100)))
    return pct


# ---------------------------------------------------------------------------
# Manifest / last-retrain lookup
# ---------------------------------------------------------------------------
def last_retrain_dates(
    instruments: Sequence[str],
    *,
    models_dir: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    base = models_dir or _models_dir()
    out: Dict[str, Optional[str]] = {}
    for inst in instruments:
        path = base / f"manifest_{inst}.json"
        out[inst] = None
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("digest: bad manifest %s: %s", path, exc)
            continue
        # The trainer writes both ``trained_at`` (ISO) and ``training_end``
        # (date). Either is fine for the human-readable digest.
        ts = data.get("trained_at") or data.get("training_end")
        if isinstance(ts, str) and len(ts) >= 10:
            out[inst] = ts[:10]
    return out


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------
@dataclass
class DigestPayload:
    date: str
    uptime_pct: int
    restarts: int
    signals: Dict[str, int]
    confidence: Counter
    last_retrain: Dict[str, Optional[str]]
    mode: str


def format_digest(payload: DigestPayload) -> str:
    uptime_emoji = "✅" if payload.uptime_pct >= 99 else ("⚠️" if payload.uptime_pct >= 90 else "🚨")

    sig_parts = " | ".join(
        f"{inst}: {payload.signals.get(inst, 0)}" for inst in payload.signals
    ) or "(none)"

    conf_parts = " | ".join(
        f"{lvl}: {payload.confidence.get(lvl, 0)}"
        for lvl in ("HIGH", "MEDIUM", "LOW")
    )

    retrain_parts = " | ".join(
        f"{inst} {dt or 'never'}" for inst, dt in payload.last_retrain.items()
    ) or "(none)"

    return (
        f"📊 *PRISM Daily — {payload.date}*\n\n"
        f"*Runner:* {uptime_emoji} Up {payload.uptime_pct}% ({payload.restarts} restarts)\n"
        f"*Signals (24h):* {sig_parts}\n"
        f"*Confidence:* {conf_parts}\n"
        f"*Last retrain:* {retrain_parts}\n"
        f"*Mode:* {payload.mode}"
    )


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------
def post_slack(text: str) -> bool:
    try:
        from prism.delivery.slack_notifier import SlackNotifier
    except Exception as exc:  # pragma: no cover
        logger.error("Could not import SlackNotifier: %s", exc)
        return False
    try:
        notifier = SlackNotifier()
        if not notifier.client:
            logger.info("Slack not configured; would have posted:\n%s", text)
            return False
        return notifier.send_alert(text) is not None
    except Exception as exc:  # pragma: no cover
        logger.error("Slack post failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging() -> Path:
    path = Path(os.environ.get("PRISM_DIGEST_LOG", DEFAULT_LOG_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve()
        for h in logger.handlers
    )
    if not has_file_handler:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        ))
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_payload(
    *,
    today: Optional[datetime] = None,
    instruments: Optional[Sequence[str]] = None,
    state_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    watchdog_log: Optional[Path] = None,
) -> DigestPayload:
    today = today or datetime.now(timezone.utc)
    insts = list(instruments or _instruments())

    counts, conf = signals_and_confidence(insts, today=today, state_dir=state_dir)
    restarts = watchdog_restarts_in_window(log_path=watchdog_log, today=today)
    retrains = last_retrain_dates(insts, models_dir=models_dir)

    return DigestPayload(
        date=today.strftime("%Y-%m-%d"),
        uptime_pct=runner_uptime_pct(restarts),
        restarts=restarts,
        signals=counts,
        confidence=conf,
        last_retrain=retrains,
        mode=_execution_mode(),
    )


def run_once(
    *,
    today: Optional[datetime] = None,
    instruments: Optional[Sequence[str]] = None,
    slack_fn=post_slack,
) -> str:
    _configure_logging()
    payload = build_payload(today=today, instruments=instruments)
    msg = format_digest(payload)
    logger.info("digest payload: restarts=%d signals=%s mode=%s",
                payload.restarts, dict(payload.signals), payload.mode)
    slack_fn(msg)
    return msg


def main(argv: Optional[Iterable[str]] = None) -> int:
    run_once()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
