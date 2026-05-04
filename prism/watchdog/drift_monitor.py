"""PRISM model-drift monitor + auto-retrain.

Runs once daily (typically 03:00 UTC, scheduled via ``PRISM-DriftMonitor``).
Inspects the per-instrument signal audit log for the trailing 7 days;
when the model is firing too rarely, leaning too hard on NEUTRAL, or
showing low mean confidence, it kicks off
``python -m prism.model.retrain --instrument <INST>``, validates the
freshly written artefacts, and (on success) restarts the runner.

Drift criteria (any one trips retrain):
  - Average daily signal count < ``PRISM_DRIFT_MIN_SIGNALS`` (default 3)
  - Fraction of NEUTRAL signals  > ``PRISM_DRIFT_NEUTRAL_PCT``  (default 0.60)
  - Mean confidence              < ``PRISM_DRIFT_MIN_CONFIDENCE`` (default 0.45)

Validation is delegated to ``prism.model.predict.missing_model_files`` —
if any of the four model layers (or the feature_cols sidecar) are missing
after retrain, the old artefacts are kept and Slack is alerted instead.

The module is import-safe on macOS / Linux for CI: ``schtasks`` and
``subprocess`` calls are isolated behind helpers that tests patch.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger("prism.drift_monitor")

DEFAULT_MIN_SIGNALS = 3.0
DEFAULT_NEUTRAL_PCT = 0.60
DEFAULT_MIN_CONFIDENCE = 0.45
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_AUDIT_ROOT = "state/signal_audit"
DEFAULT_LOG_PATH = "logs/drift_monitor.log"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _instruments() -> List[str]:
    raw = os.environ.get("PRISM_INSTRUMENTS", "XAUUSD,EURUSD,GBPUSD")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _thresholds() -> dict:
    return {
        "min_signals": float(os.environ.get("PRISM_DRIFT_MIN_SIGNALS", DEFAULT_MIN_SIGNALS)),
        "neutral_pct": float(os.environ.get("PRISM_DRIFT_NEUTRAL_PCT", DEFAULT_NEUTRAL_PCT)),
        "min_confidence": float(os.environ.get("PRISM_DRIFT_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)),
        "lookback_days": int(os.environ.get("PRISM_DRIFT_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)),
    }


def _audit_root() -> Path:
    state_dir = os.environ.get("PRISM_STATE_DIR", "state")
    return Path(state_dir) / "signal_audit"


# ---------------------------------------------------------------------------
# Audit log loading
# ---------------------------------------------------------------------------
@dataclass
class DriftStats:
    instrument: str
    days_with_data: int
    total_signals: int
    avg_signals_per_day: float
    neutral_pct: float
    mean_confidence: float
    drifted: bool
    reasons: List[str]


def _load_audit_records(
    instrument: str,
    *,
    lookback_days: int,
    today: Optional[datetime] = None,
    audit_root: Optional[Path] = None,
) -> List[dict]:
    """Read the trailing ``lookback_days`` of audit JSONL for an instrument.

    Missing daily files are silently skipped (the runner doesn't write a
    file when nothing fires that day). Bad lines are dropped with a log.
    """
    root = audit_root or _audit_root()
    inst_dir = root / instrument
    if not inst_dir.exists():
        return []

    today = today or datetime.now(timezone.utc)
    out: List[dict] = []
    for offset in range(lookback_days):
        day = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        path = inst_dir / f"{day}.jsonl"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "drift: skipping malformed line in %s:%d — %s",
                            path, lineno, exc,
                        )
        except OSError as exc:
            logger.error("drift: could not read %s: %s", path, exc)
    return out


def compute_drift_stats(
    instrument: str,
    *,
    thresholds: Optional[dict] = None,
    today: Optional[datetime] = None,
    audit_root: Optional[Path] = None,
) -> DriftStats:
    """Compute drift indicators for one instrument over the lookback window."""
    th = thresholds or _thresholds()
    records = _load_audit_records(
        instrument,
        lookback_days=th["lookback_days"],
        today=today,
        audit_root=audit_root,
    )

    days_seen: set[str] = set()
    neutral = 0
    confs: List[float] = []
    for r in records:
        ts = r.get("audit_ts") or r.get("signal_time") or ""
        if isinstance(ts, str) and len(ts) >= 10:
            days_seen.add(ts[:10])
        direction = (r.get("direction") or "").upper()
        if direction == "NEUTRAL":
            neutral += 1
        try:
            confs.append(float(r.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue

    total = len(records)
    days_with_data = max(len(days_seen), 1)  # avoid div-by-zero on empty days
    avg_per_day = total / float(th["lookback_days"])  # daily-average over WINDOW
    neutral_pct = (neutral / total) if total else 0.0
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0

    reasons: List[str] = []
    if avg_per_day < th["min_signals"]:
        reasons.append(
            f"avg signals/day {avg_per_day:.2f} < {th['min_signals']}"
        )
    if total > 0 and neutral_pct > th["neutral_pct"]:
        reasons.append(
            f"NEUTRAL share {neutral_pct:.0%} > {th['neutral_pct']:.0%}"
        )
    if total > 0 and mean_conf < th["min_confidence"]:
        reasons.append(
            f"mean confidence {mean_conf:.2f} < {th['min_confidence']:.2f}"
        )

    return DriftStats(
        instrument=instrument,
        days_with_data=len(days_seen),
        total_signals=total,
        avg_signals_per_day=avg_per_day,
        neutral_pct=neutral_pct,
        mean_confidence=mean_conf,
        drifted=bool(reasons),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Retrain + validation
# ---------------------------------------------------------------------------
def run_retrain(instrument: str) -> int:
    """Run ``python -m prism.model.retrain --instrument <INST>``."""
    cmd = [sys.executable, "-m", "prism.model.retrain", "--instrument", instrument]
    logger.info("drift: running retrain: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 60)
        if result.returncode != 0:
            logger.error(
                "retrain %s failed rc=%d stderr=%s",
                instrument, result.returncode, (result.stderr or "")[-500:],
            )
        return result.returncode
    except subprocess.SubprocessError as exc:
        logger.error("retrain subprocess failed: %s", exc)
        return 1


def validate_artefacts(instruments: Sequence[str]) -> List[Path]:
    """Wrap ``predict.missing_model_files`` so tests can patch one symbol."""
    from prism.model.predict import missing_model_files

    return list(missing_model_files(list(instruments)))


def restart_runner(task_name: str = "PRISM-Runner") -> bool:
    """Stop+start the runner via schtasks. Returns True on success."""
    try:
        subprocess.run(
            ["schtasks", "/end", "/tn", task_name],
            capture_output=True, text=True, timeout=30, check=False,
        )
        result = subprocess.run(
            ["schtasks", "/run", "/tn", task_name],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.error("runner restart failed: %s", exc)
        return False


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
            logger.info("Slack not configured; would have posted: %s", text)
            return False
        return notifier.send_alert(text) is not None
    except Exception as exc:  # pragma: no cover
        logger.error("Slack post failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging() -> Path:
    path = Path(os.environ.get("PRISM_DRIFT_LOG", DEFAULT_LOG_PATH))
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
def process_instrument(
    instrument: str,
    *,
    thresholds: Optional[dict] = None,
    today: Optional[datetime] = None,
    audit_root: Optional[Path] = None,
    retrain_fn=run_retrain,
    validate_fn=validate_artefacts,
    restart_fn=restart_runner,
    slack_fn=post_slack,
) -> dict:
    """Decide + (optionally) retrain + (on success) restart the runner.

    Returns a result dict suitable for digest summaries / tests.
    """
    stats = compute_drift_stats(
        instrument, thresholds=thresholds, today=today, audit_root=audit_root,
    )
    result = {
        "instrument": instrument,
        "drifted": stats.drifted,
        "reasons": list(stats.reasons),
        "retrained": False,
        "validation_failed": False,
        "runner_restarted": False,
    }
    if not stats.drifted:
        logger.info("drift: %s healthy — %s", instrument, _stats_summary(stats))
        return result

    logger.warning(
        "drift: %s drift detected (%s); retraining",
        instrument, "; ".join(stats.reasons),
    )
    rc = retrain_fn(instrument)
    if rc != 0:
        slack_fn(
            f"⚠️ PRISM auto-retrain FAILED for {instrument} "
            f"(rc={rc}). Old models still live."
        )
        return result
    result["retrained"] = True

    missing = validate_fn([instrument])
    if missing:
        result["validation_failed"] = True
        slack_fn(
            f"⚠️ PRISM auto-retrain for {instrument} produced incomplete "
            f"artefacts ({len(missing)} missing). Old models still live."
        )
        return result

    if restart_fn():
        result["runner_restarted"] = True
    headline_reason = stats.reasons[0] if stats.reasons else "drift detected"
    slack_fn(
        f"🧠 PRISM auto-retrained {instrument} (drift detected: "
        f"{headline_reason}). Runner restarted with fresh models."
    )
    return result


def _stats_summary(stats: DriftStats) -> str:
    return (
        f"signals={stats.total_signals} avg/day={stats.avg_signals_per_day:.2f} "
        f"neutral_pct={stats.neutral_pct:.0%} mean_conf={stats.mean_confidence:.2f}"
    )


def run_once(
    *,
    instruments: Optional[Sequence[str]] = None,
    thresholds: Optional[dict] = None,
    today: Optional[datetime] = None,
) -> List[dict]:
    _configure_logging()
    insts = list(instruments or _instruments())
    th = thresholds or _thresholds()
    logger.info(
        "drift: starting check for %d instruments thresholds=%s",
        len(insts), th,
    )
    return [process_instrument(i, thresholds=th, today=today) for i in insts]


def main(argv: Optional[Iterable[str]] = None) -> int:
    run_once()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
