"""Tests for prism.watchdog.watchdog.

The watchdog is a long-running supervisor that talks to ``schtasks`` and
Slack. We mock both so the suite is portable and deterministic. Coverage:

  - schtasks_run delegates to subprocess and returns its rc
  - runner_is_running honours psutil's view of the process table
  - attempt_restart triggers schtasks, sleeps, re-checks, returns alive bit
  - handle_runner_down counts attempts, sleeps between them, posts the
    correct Slack message on success vs. terminal failure
  - run_forever invokes the down-handler when the runner is missing and
    skips it when alive
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prism.watchdog import watchdog as wd  # noqa: E402


# ---------------------------------------------------------------------------
# schtasks_run
# ---------------------------------------------------------------------------
def test_schtasks_run_returns_subprocess_rc():
    fake = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("prism.watchdog.watchdog.subprocess.run", return_value=fake) as run:
        rc = wd.schtasks_run("PRISM-Runner")
    assert rc == 0
    args = run.call_args[0][0]
    assert args == ["schtasks", "/run", "/tn", "PRISM-Runner"]


def test_schtasks_run_on_subprocess_error_returns_1():
    with patch("prism.watchdog.watchdog.subprocess.run",
               side_effect=FileNotFoundError("no schtasks")):
        rc = wd.schtasks_run("PRISM-Runner")
    assert rc == 1


# ---------------------------------------------------------------------------
# runner_is_running
# ---------------------------------------------------------------------------
def test_runner_is_running_finds_match_via_psutil():
    fake_proc = MagicMock()
    fake_proc.info = {"name": "python.exe"}
    fake_psutil = MagicMock()
    fake_psutil.process_iter.return_value = [fake_proc]
    fake_psutil.NoSuchProcess = Exception
    fake_psutil.AccessDenied = Exception
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert wd.runner_is_running("python.exe") is True


def test_runner_is_running_returns_false_when_absent():
    fake_psutil = MagicMock()
    fake_psutil.process_iter.return_value = []
    fake_psutil.NoSuchProcess = Exception
    fake_psutil.AccessDenied = Exception
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert wd.runner_is_running("python.exe") is False


# ---------------------------------------------------------------------------
# attempt_restart
# ---------------------------------------------------------------------------
def test_attempt_restart_returns_true_when_alive_after_wait():
    sleeps: list = []
    with patch("prism.watchdog.watchdog.schtasks_run", return_value=0) as srun, \
         patch("prism.watchdog.watchdog.runner_is_running", return_value=True):
        ok = wd.attempt_restart(verify_sec=15, sleep_fn=sleeps.append)
    assert ok is True
    assert sleeps == [15]
    srun.assert_called_once()


def test_attempt_restart_returns_false_when_still_dead():
    with patch("prism.watchdog.watchdog.schtasks_run", return_value=0), \
         patch("prism.watchdog.watchdog.runner_is_running", return_value=False):
        ok = wd.attempt_restart(verify_sec=0, sleep_fn=lambda *_: None)
    assert ok is False


# ---------------------------------------------------------------------------
# handle_runner_down — escalation
# ---------------------------------------------------------------------------
def test_handle_runner_down_first_attempt_succeeds():
    slack = MagicMock(return_value=True)
    sleeps: list = []
    ok = wd.handle_runner_down(
        max_attempts=3,
        retry_sec=300,
        sleep_fn=sleeps.append,
        restart_fn=MagicMock(return_value=True),
        slack_fn=slack,
    )
    assert ok is True
    # No retry sleeps when we win on attempt 1.
    assert sleeps == []
    msg = slack.call_args[0][0]
    assert "back online" in msg
    assert "🔄" in msg


def test_handle_runner_down_recovers_after_retries():
    slack = MagicMock()
    sleeps: list = []
    restart = MagicMock(side_effect=[False, False, True])
    ok = wd.handle_runner_down(
        max_attempts=3,
        retry_sec=300,
        sleep_fn=sleeps.append,
        restart_fn=restart,
        slack_fn=slack,
    )
    assert ok is True
    assert restart.call_count == 3
    # Two retry sleeps between three attempts.
    assert sleeps == [300, 300]
    assert "back online" in slack.call_args[0][0]


def test_handle_runner_down_gives_up_after_max_attempts():
    slack = MagicMock()
    sleeps: list = []
    restart = MagicMock(return_value=False)
    ok = wd.handle_runner_down(
        max_attempts=3,
        retry_sec=300,
        sleep_fn=sleeps.append,
        restart_fn=restart,
        slack_fn=slack,
    )
    assert ok is False
    assert restart.call_count == 3
    msg = slack.call_args[0][0]
    assert "Manual intervention" in msg
    assert "🚨" in msg


# ---------------------------------------------------------------------------
# run_forever
# ---------------------------------------------------------------------------
def test_run_forever_calls_down_handler_when_runner_missing():
    on_down = MagicMock()
    runner_states = iter([False, True])
    with patch("prism.watchdog.watchdog._configure_logging", return_value=Path("/tmp/x.log")):
        wd.run_forever(
            check_sec=0,
            max_iterations=2,
            sleep_fn=lambda *_: None,
            runner_check_fn=lambda: next(runner_states),
            on_down=on_down,
        )
    # First iteration sees False -> on_down; second sees True -> skip.
    assert on_down.call_count == 1


def test_run_forever_skips_handler_when_runner_alive():
    on_down = MagicMock()
    with patch("prism.watchdog.watchdog._configure_logging", return_value=Path("/tmp/x.log")):
        wd.run_forever(
            check_sec=0,
            max_iterations=3,
            sleep_fn=lambda *_: None,
            runner_check_fn=lambda: True,
            on_down=on_down,
        )
    on_down.assert_not_called()


# ---------------------------------------------------------------------------
# post_slack
# ---------------------------------------------------------------------------
def test_post_slack_uses_slack_notifier(monkeypatch):
    fake_notifier = MagicMock()
    fake_notifier.client = MagicMock()  # truthy
    fake_notifier.send_alert.return_value = "1234.5"
    fake_class = MagicMock(return_value=fake_notifier)
    fake_module = MagicMock(SlackNotifier=fake_class)
    monkeypatch.setitem(sys.modules, "prism.delivery.slack_notifier", fake_module)

    assert wd.post_slack("hello") is True
    fake_notifier.send_alert.assert_called_once_with("hello")


def test_post_slack_returns_false_when_no_client(monkeypatch):
    fake_notifier = MagicMock()
    fake_notifier.client = None
    fake_class = MagicMock(return_value=fake_notifier)
    fake_module = MagicMock(SlackNotifier=fake_class)
    monkeypatch.setitem(sys.modules, "prism.delivery.slack_notifier", fake_module)

    assert wd.post_slack("hello") is False
