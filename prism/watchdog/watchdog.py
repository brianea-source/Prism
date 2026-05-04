"""PRISM runner watchdog.

Lightweight long-running loop that lives next to the live runner on the
Windows VPS. Polls the runner every ``CHECK_INTERVAL_SEC`` (default 300s,
i.e. 5 minutes); when the runner process is missing it tries to restart
the ``PRISM-Runner`` scheduled task up to ``MAX_RESTART_ATTEMPTS`` times
(default 3, spaced 5 minutes apart). Every state transition — successful
restart, retry, give-up — is mirrored to Slack and to ``logs/watchdog.log``.

Design constraints:
  - **Separate process** from the runner. A bug in the runner must not
    take the watchdog down with it (and vice-versa).
  - **Failure-tolerant Slack**. If Slack is unreachable we still log
    locally and continue; the alerting layer can never block recovery.
  - **Idempotent restart logic**. If the runner is already healthy when
    we go to restart it, that's fine — ``schtasks /run`` is a no-op when
    the task is already running.
  - **PID-file based detection**. The runner writes ``state/runner.pid``
    on startup and removes it on clean shutdown. The watchdog reads this
    PID and checks whether the process is alive via ``psutil.pid_exists``
    (or ``os.kill(pid, 0)`` as fallback). This avoids false positives
    from other Python processes (retrain, drift monitor, pip, etc.).
  - **Cross-platform-ish**. Restarts are Windows-only via ``schtasks``;
    the helpers shell out so tests can patch ``subprocess.run``.

Configuration (env vars, all optional):
  - ``PRISM_WATCHDOG_TASK_NAME``     (default ``PRISM-Runner``)
  - ``PRISM_STATE_DIR``              (default ``state`` — PID file lives here)
  - ``PRISM_WATCHDOG_CHECK_SEC``     (default ``300``)
  - ``PRISM_WATCHDOG_VERIFY_SEC``    (default ``15``)
  - ``PRISM_WATCHDOG_RETRY_SEC``     (default ``300``)
  - ``PRISM_WATCHDOG_MAX_ATTEMPTS``  (default ``3``)
  - ``PRISM_WATCHDOG_LOG``           (default ``logs/watchdog.log``)
  - ``PRISM_SLACK_TOKEN`` / ``PRISM_SLACK_CHANNEL`` — reuses the runner's
    Slack app credentials. When either is unset, alerts are logged locally
    only (no crash).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("prism.watchdog")

# ---------------------------------------------------------------------------
# Defaults — kept module-level so tests can monkeypatch them.
# ---------------------------------------------------------------------------
DEFAULT_TASK_NAME = "PRISM-Runner"
DEFAULT_CHECK_SEC = 300        # 5 min between health checks
DEFAULT_VERIFY_SEC = 15        # wait after schtasks /run before re-checking
DEFAULT_RETRY_SEC = 300        # 5 min between restart attempts
DEFAULT_MAX_ATTEMPTS = 3       # then escalate
DEFAULT_LOG_PATH = "logs/watchdog.log"
DEFAULT_PID_FILE = "state/runner.pid"


# ---------------------------------------------------------------------------
# Process detection via PID file
# ---------------------------------------------------------------------------
def _runner_task_name() -> str:
    return os.environ.get("PRISM_WATCHDOG_TASK_NAME", DEFAULT_TASK_NAME)


def _pid_file_path() -> Path:
    state_dir = os.environ.get("PRISM_STATE_DIR", "state")
    return Path(state_dir) / "runner.pid"


def _pid_is_alive(pid: int) -> bool:
    """Check whether a process with the given PID exists."""
    try:
        import psutil  # type: ignore
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    # Fallback: os.kill with signal 0 probes without killing.
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


def runner_is_running(pid_path: Optional[Path] = None) -> bool:
    """Return True when the runner's PID file exists and the process is alive.

    This is precise — unlike scanning for ``python.exe``, it won't
    false-positive on retrain, drift monitor, pip, or any other Python
    process running on the same host.
    """
    path = pid_path or _pid_file_path()
    if not path.exists():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as exc:
        logger.warning("Could not read PID file %s: %s", path, exc)
        return False
    alive = _pid_is_alive(pid)
    if not alive:
        logger.info("PID %d from %s is not alive — runner is down", pid, path)
    return alive


# ---------------------------------------------------------------------------
# Scheduled-task control (Windows ``schtasks``)
# ---------------------------------------------------------------------------
def schtasks_run(task_name: Optional[str] = None) -> int:
    """Trigger ``schtasks /run /tn <task>``. Returns the exit code."""
    name = task_name or _runner_task_name()
    try:
        result = subprocess.run(
            ["schtasks", "/run", "/tn", name],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "schtasks /run /tn %s exit=%d stdout=%r stderr=%r",
                name, result.returncode, result.stdout, result.stderr,
            )
        return result.returncode
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.error("schtasks /run failed for %s: %s", name, exc)
        return 1


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------
def post_slack(text: str) -> bool:
    """Best-effort Slack post. Returns True on success.

    Reuses the runner's :class:`~prism.delivery.slack_notifier.SlackNotifier`
    so we don't double-import slack_sdk and so credential plumbing is
    identical. Any failure is swallowed (logged); the watchdog must never
    crash because Slack is down.
    """
    try:
        from prism.delivery.slack_notifier import SlackNotifier
    except Exception as exc:  # pragma: no cover — import diagnostics
        logger.error("Could not import SlackNotifier: %s", exc)
        return False
    try:
        notifier = SlackNotifier()
        if not notifier.client:
            logger.info("Slack not configured; would have posted: %s", text)
            return False
        ts = notifier.send_alert(text)
        return ts is not None
    except Exception as exc:  # pragma: no cover
        logger.error("Slack post failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
def _configure_logging(log_path: Optional[str] = None) -> Path:
    path = Path(log_path or os.environ.get("PRISM_WATCHDOG_LOG", DEFAULT_LOG_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Idempotent: don't double-attach handlers if the loop runs twice in
    # the same process (tests, REPL).
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
# Restart workflow
# ---------------------------------------------------------------------------
def attempt_restart(
    *,
    task_name: Optional[str] = None,
    verify_sec: Optional[int] = None,
    sleep_fn=time.sleep,
) -> bool:
    """One restart attempt: trigger schtasks, wait, re-check liveness.

    Returns True if the runner is alive after the verification window.
    """
    task = task_name or _runner_task_name()
    wait = verify_sec if verify_sec is not None else int(
        os.environ.get("PRISM_WATCHDOG_VERIFY_SEC", DEFAULT_VERIFY_SEC)
    )

    rc = schtasks_run(task)
    logger.info("schtasks /run /tn %s -> rc=%d; waiting %ds", task, rc, wait)
    sleep_fn(wait)
    alive = runner_is_running()
    logger.info("post-restart check: runner alive=%s", alive)
    return alive


def handle_runner_down(
    *,
    max_attempts: Optional[int] = None,
    retry_sec: Optional[int] = None,
    sleep_fn=time.sleep,
    restart_fn=attempt_restart,
    slack_fn=post_slack,
) -> bool:
    """Run the full restart escalation. Returns True if recovery succeeded."""
    attempts = max_attempts if max_attempts is not None else int(
        os.environ.get("PRISM_WATCHDOG_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)
    )
    delay = retry_sec if retry_sec is not None else int(
        os.environ.get("PRISM_WATCHDOG_RETRY_SEC", DEFAULT_RETRY_SEC)
    )

    for i in range(1, attempts + 1):
        logger.warning("Runner down — restart attempt %d/%d", i, attempts)
        if restart_fn():
            msg = "🔄 PRISM runner restarted automatically — back online."
            logger.info("Recovery succeeded on attempt %d/%d", i, attempts)
            slack_fn(msg)
            return True
        if i < attempts:
            logger.info("Attempt %d failed; sleeping %ds before retry", i, delay)
            sleep_fn(delay)

    msg = "🚨 PRISM runner down and could not be restarted. Manual intervention needed."
    logger.error("All %d restart attempts failed", attempts)
    slack_fn(msg)
    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_forever(
    *,
    check_sec: Optional[int] = None,
    max_iterations: Optional[int] = None,
    sleep_fn=time.sleep,
    runner_check_fn=runner_is_running,
    on_down=handle_runner_down,
) -> None:
    """Watchdog main loop.

    ``max_iterations`` is for tests (unbounded by default).
    """
    interval = check_sec if check_sec is not None else int(
        os.environ.get("PRISM_WATCHDOG_CHECK_SEC", DEFAULT_CHECK_SEC)
    )
    log_path = _configure_logging()
    logger.info(
        "Watchdog started — task=%s pid_file=%s interval=%ds log=%s",
        _runner_task_name(), _pid_file_path(), interval, log_path,
    )

    iters = 0
    while True:
        try:
            if not runner_check_fn():
                on_down()
            else:
                logger.debug("Runner healthy")
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("Watchdog iteration crashed: %s", exc)

        iters += 1
        if max_iterations is not None and iters >= max_iterations:
            return
        sleep_fn(interval)


def main(argv: Optional[Iterable[str]] = None) -> int:
    _configure_logging()
    try:
        run_forever()
    except KeyboardInterrupt:
        logger.info("Watchdog stopped via SIGINT")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
