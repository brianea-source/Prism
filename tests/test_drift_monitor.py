"""Tests for prism.watchdog.drift_monitor.

We seed JSONL audit files into a tmp directory and assert the threshold
math + retrain orchestration. No real subprocess calls; ``run_retrain``
and ``restart_runner`` are patched.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prism.watchdog import drift_monitor as dm  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_audit(
    root: Path,
    instrument: str,
    day: datetime,
    records: list[dict],
) -> None:
    inst_dir = root / instrument
    inst_dir.mkdir(parents=True, exist_ok=True)
    path = inst_dir / f"{day.strftime('%Y-%m-%d')}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _rec(direction="LONG", confidence=0.7, ts="2026-05-04T12:00:00+00:00"):
    return {
        "audit_ts": ts,
        "instrument": "XAUUSD",
        "direction": direction,
        "confidence": confidence,
        "confidence_level": "HIGH",
    }


# ---------------------------------------------------------------------------
# compute_drift_stats
# ---------------------------------------------------------------------------
def test_drift_stats_clean_window_no_drift(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    # 7 days x 5 LONG signals/day @ 0.7 confidence — well above all thresholds.
    for offset in range(7):
        day = today - timedelta(days=offset)
        _write_audit(
            tmp_path, "XAUUSD", day,
            [_rec(ts=day.isoformat()) for _ in range(5)],
        )

    stats = dm.compute_drift_stats("XAUUSD", today=today, audit_root=tmp_path)
    assert stats.drifted is False
    assert stats.total_signals == 35
    assert stats.avg_signals_per_day == pytest.approx(5.0)
    assert stats.neutral_pct == 0.0
    assert stats.mean_confidence == pytest.approx(0.7)


def test_drift_stats_too_few_signals(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    # 7 signals total over 7 days → 1/day < threshold of 3.
    for offset in range(7):
        day = today - timedelta(days=offset)
        _write_audit(tmp_path, "XAUUSD", day, [_rec(ts=day.isoformat())])

    stats = dm.compute_drift_stats("XAUUSD", today=today, audit_root=tmp_path)
    assert stats.drifted is True
    assert any("avg signals/day" in r for r in stats.reasons)


def test_drift_stats_too_many_neutral(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    # 5 signals/day, 70% NEUTRAL.
    for offset in range(7):
        day = today - timedelta(days=offset)
        recs = [
            _rec(direction="NEUTRAL", ts=day.isoformat()) for _ in range(7)
        ] + [_rec(direction="LONG", ts=day.isoformat()) for _ in range(3)]
        _write_audit(tmp_path, "XAUUSD", day, recs)

    stats = dm.compute_drift_stats("XAUUSD", today=today, audit_root=tmp_path)
    assert stats.drifted is True
    assert any("NEUTRAL share" in r for r in stats.reasons)


def test_drift_stats_low_confidence(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    for offset in range(7):
        day = today - timedelta(days=offset)
        _write_audit(
            tmp_path, "XAUUSD", day,
            [_rec(confidence=0.30, ts=day.isoformat()) for _ in range(5)],
        )

    stats = dm.compute_drift_stats("XAUUSD", today=today, audit_root=tmp_path)
    assert stats.drifted is True
    assert any("mean confidence" in r for r in stats.reasons)


def test_drift_stats_handles_missing_files(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    # No files written at all.
    stats = dm.compute_drift_stats("XAUUSD", today=today, audit_root=tmp_path)
    # Empty audit → 0 signals/day → drifted on min_signals.
    assert stats.total_signals == 0
    assert stats.drifted is True
    assert any("avg signals/day" in r for r in stats.reasons)


def test_drift_stats_skips_malformed_lines(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    inst_dir = tmp_path / "XAUUSD"
    inst_dir.mkdir(parents=True)
    path = inst_dir / f"{today.strftime('%Y-%m-%d')}.jsonl"
    path.write_text(
        "this is not json\n"
        + json.dumps(_rec(ts=today.isoformat())) + "\n"
        + "{broken json\n",
        encoding="utf-8",
    )
    stats = dm.compute_drift_stats("XAUUSD", today=today, audit_root=tmp_path)
    assert stats.total_signals == 1


# ---------------------------------------------------------------------------
# process_instrument — orchestration
# ---------------------------------------------------------------------------
def test_process_instrument_no_drift_skips_retrain(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    for offset in range(7):
        day = today - timedelta(days=offset)
        _write_audit(
            tmp_path, "XAUUSD", day,
            [_rec(ts=day.isoformat()) for _ in range(5)],
        )

    retrain = MagicMock(return_value=0)
    validate = MagicMock(return_value=[])
    restart = MagicMock(return_value=True)
    slack = MagicMock(return_value=True)
    out = dm.process_instrument(
        "XAUUSD",
        today=today,
        audit_root=tmp_path,
        retrain_fn=retrain,
        validate_fn=validate,
        restart_fn=restart,
        slack_fn=slack,
    )
    assert out["drifted"] is False
    retrain.assert_not_called()
    restart.assert_not_called()
    slack.assert_not_called()


def test_process_instrument_drift_triggers_retrain_and_restart(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    # Drifted: too few signals.
    _write_audit(tmp_path, "XAUUSD", today, [_rec(ts=today.isoformat())])

    retrain = MagicMock(return_value=0)
    validate = MagicMock(return_value=[])
    restart = MagicMock(return_value=True)
    slack = MagicMock(return_value=True)
    out = dm.process_instrument(
        "XAUUSD",
        today=today,
        audit_root=tmp_path,
        retrain_fn=retrain,
        validate_fn=validate,
        restart_fn=restart,
        slack_fn=slack,
    )
    assert out["drifted"] is True
    assert out["retrained"] is True
    assert out["validation_failed"] is False
    assert out["runner_restarted"] is True
    retrain.assert_called_once_with("XAUUSD")
    restart.assert_called_once()
    msg = slack.call_args[0][0]
    assert "auto-retrained XAUUSD" in msg


def test_process_instrument_aborts_on_validation_failure(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    _write_audit(tmp_path, "XAUUSD", today, [_rec(ts=today.isoformat())])

    retrain = MagicMock(return_value=0)
    validate = MagicMock(return_value=[Path("models/missing.joblib")])
    restart = MagicMock(return_value=True)
    slack = MagicMock(return_value=True)
    out = dm.process_instrument(
        "XAUUSD",
        today=today,
        audit_root=tmp_path,
        retrain_fn=retrain,
        validate_fn=validate,
        restart_fn=restart,
        slack_fn=slack,
    )
    assert out["validation_failed"] is True
    assert out["runner_restarted"] is False
    restart.assert_not_called()
    msg = slack.call_args[0][0]
    assert "incomplete artefacts" in msg


def test_process_instrument_aborts_on_retrain_failure(tmp_path):
    today = datetime(2026, 5, 4, tzinfo=timezone.utc)
    _write_audit(tmp_path, "XAUUSD", today, [_rec(ts=today.isoformat())])

    retrain = MagicMock(return_value=1)
    validate = MagicMock()
    restart = MagicMock()
    slack = MagicMock()
    out = dm.process_instrument(
        "XAUUSD",
        today=today,
        audit_root=tmp_path,
        retrain_fn=retrain,
        validate_fn=validate,
        restart_fn=restart,
        slack_fn=slack,
    )
    assert out["retrained"] is False
    validate.assert_not_called()
    restart.assert_not_called()
    msg = slack.call_args[0][0]
    assert "FAILED" in msg


# ---------------------------------------------------------------------------
# run_retrain / restart_runner — wrappers
# ---------------------------------------------------------------------------
def test_run_retrain_invokes_subprocess():
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch("prism.watchdog.drift_monitor.subprocess.run", return_value=fake) as run:
        rc = dm.run_retrain("EURUSD")
    assert rc == 0
    args = run.call_args[0][0]
    assert "prism.model.retrain" in args
    assert "EURUSD" in args


def test_restart_runner_runs_end_then_run():
    fake_end = MagicMock(returncode=0)
    fake_run = MagicMock(returncode=0)
    with patch("prism.watchdog.drift_monitor.subprocess.run",
               side_effect=[fake_end, fake_run]) as run:
        ok = dm.restart_runner("PRISM-Runner")
    assert ok is True
    calls = [c.args[0] for c in run.call_args_list]
    assert calls[0][:3] == ["schtasks", "/end", "/tn"]
    assert calls[1][:3] == ["schtasks", "/run", "/tn"]
