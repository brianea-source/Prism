"""Tests for prism.watchdog.daily_digest."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prism.watchdog import daily_digest as dd  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# signals_and_confidence
# ---------------------------------------------------------------------------
def test_signals_and_confidence_counts_per_instrument(tmp_path):
    today = datetime(2026, 5, 5, 8, 0, tzinfo=timezone.utc)
    audit_root = tmp_path / "signal_audit"
    _write_jsonl(
        audit_root / "XAUUSD" / "2026-05-05.jsonl",
        [
            {"audit_ts": today.isoformat(), "direction": "LONG",
             "confidence_level": "HIGH"},
            {"audit_ts": today.isoformat(), "direction": "LONG",
             "confidence_level": "MEDIUM"},
            {"audit_ts": today.isoformat(), "direction": "NEUTRAL",
             "confidence_level": "LOW"},
        ],
    )
    _write_jsonl(
        audit_root / "EURUSD" / "2026-05-05.jsonl",
        [
            {"audit_ts": today.isoformat(), "direction": "SHORT",
             "confidence_level": "MEDIUM"},
            {"audit_ts": today.isoformat(), "direction": "SHORT",
             "confidence_level": "MEDIUM"},
        ],
    )

    counts, conf = dd.signals_and_confidence(
        ["XAUUSD", "EURUSD"], today=today, state_dir=tmp_path,
    )
    assert counts == {"XAUUSD": 3, "EURUSD": 2}
    assert conf["HIGH"] == 1
    assert conf["MEDIUM"] == 3
    assert conf["LOW"] == 1


def test_signals_and_confidence_handles_missing_file(tmp_path):
    today = datetime(2026, 5, 5, tzinfo=timezone.utc)
    counts, conf = dd.signals_and_confidence(
        ["XAUUSD"], today=today, state_dir=tmp_path,
    )
    assert counts == {"XAUUSD": 0}
    assert conf == {}


# ---------------------------------------------------------------------------
# watchdog_restarts_in_window
# ---------------------------------------------------------------------------
def test_watchdog_restarts_counted_in_window(tmp_path):
    log_path = tmp_path / "watchdog.log"
    today = datetime(2026, 5, 5, 8, 0, tzinfo=timezone.utc)
    in_window = (today - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
    out_window = (today - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%S")
    log_path.write_text(
        f"{in_window}+0000 [INFO] prism.watchdog — Recovery succeeded on attempt 1/3\n"
        f"{in_window}+0000 [INFO] prism.watchdog — Watchdog started\n"
        f"{out_window}+0000 [INFO] prism.watchdog — Recovery succeeded on attempt 2/3\n",
        encoding="utf-8",
    )
    n = dd.watchdog_restarts_in_window(log_path=log_path, today=today)
    assert n == 1


def test_watchdog_restarts_zero_when_log_missing(tmp_path):
    n = dd.watchdog_restarts_in_window(
        log_path=tmp_path / "absent.log",
        today=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )
    assert n == 0


def test_runner_uptime_pct_perfect_and_degraded():
    assert dd.runner_uptime_pct(0) == 100
    assert dd.runner_uptime_pct(1) == 100  # 5 min lost ~= 0% rounded
    # 50 restarts × 5 min = 250 min ≈ 17% loss → 83% uptime
    assert dd.runner_uptime_pct(50) == 83


# ---------------------------------------------------------------------------
# last_retrain_dates
# ---------------------------------------------------------------------------
def test_last_retrain_reads_manifest_trained_at(tmp_path):
    (tmp_path / "manifest_XAUUSD.json").write_text(
        json.dumps({"trained_at": "2026-05-04T12:34:56+00:00"}),
        encoding="utf-8",
    )
    (tmp_path / "manifest_EURUSD.json").write_text(
        json.dumps({"training_end": "2026-05-03"}),
        encoding="utf-8",
    )
    out = dd.last_retrain_dates(["XAUUSD", "EURUSD", "GBPUSD"], models_dir=tmp_path)
    assert out["XAUUSD"] == "2026-05-04"
    assert out["EURUSD"] == "2026-05-03"
    assert out["GBPUSD"] is None


def test_last_retrain_handles_corrupt_manifest(tmp_path):
    (tmp_path / "manifest_XAUUSD.json").write_text("{bad json", encoding="utf-8")
    out = dd.last_retrain_dates(["XAUUSD"], models_dir=tmp_path)
    assert out == {"XAUUSD": None}


# ---------------------------------------------------------------------------
# format_digest
# ---------------------------------------------------------------------------
def test_format_digest_matches_spec_format():
    payload = dd.DigestPayload(
        date="2026-05-05",
        uptime_pct=100,
        restarts=0,
        signals={"XAUUSD": 3, "EURUSD": 2},
        confidence=__import__("collections").Counter({"HIGH": 1, "MEDIUM": 3, "LOW": 1}),
        last_retrain={"XAUUSD": "2026-05-04", "EURUSD": "2026-05-04"},
        mode="NOTIFY",
    )
    msg = dd.format_digest(payload)
    assert "📊 *PRISM Daily — 2026-05-05*" in msg
    assert "*Runner:* ✅ Up 100% (0 restarts)" in msg
    assert "*Signals (24h):* XAUUSD: 3 | EURUSD: 2" in msg
    assert "*Confidence:* HIGH: 1 | MEDIUM: 3 | LOW: 1" in msg
    assert "*Last retrain:* XAUUSD 2026-05-04 | EURUSD 2026-05-04" in msg
    assert "*Mode:* NOTIFY" in msg


def test_format_digest_degraded_uptime_emoji():
    from collections import Counter
    payload = dd.DigestPayload(
        date="2026-05-05",
        uptime_pct=85,
        restarts=18,
        signals={"XAUUSD": 0},
        confidence=Counter(),
        last_retrain={"XAUUSD": None},
        mode="NOTIFY",
    )
    msg = dd.format_digest(payload)
    assert "🚨" in msg
    assert "Up 85%" in msg
    assert "XAUUSD never" in msg


# ---------------------------------------------------------------------------
# build_payload + run_once
# ---------------------------------------------------------------------------
def test_build_payload_integrates_inputs(tmp_path, monkeypatch):
    today = datetime(2026, 5, 5, 8, 0, tzinfo=timezone.utc)
    state_dir = tmp_path / "state"
    audit_root = state_dir / "signal_audit"
    _write_jsonl(
        audit_root / "XAUUSD" / "2026-05-05.jsonl",
        [{"audit_ts": today.isoformat(), "direction": "LONG",
          "confidence_level": "HIGH"}],
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest_XAUUSD.json").write_text(
        json.dumps({"trained_at": "2026-05-04T00:00:00+00:00"}),
        encoding="utf-8",
    )
    log_path = tmp_path / "watchdog.log"
    log_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("PRISM_EXECUTION_MODE", "NOTIFY")

    payload = dd.build_payload(
        today=today,
        instruments=["XAUUSD"],
        state_dir=state_dir,
        models_dir=models_dir,
        watchdog_log=log_path,
    )
    assert payload.signals == {"XAUUSD": 1}
    assert payload.confidence["HIGH"] == 1
    assert payload.last_retrain["XAUUSD"] == "2026-05-04"
    assert payload.restarts == 0
    assert payload.uptime_pct == 100
    assert payload.mode == "NOTIFY"


def test_run_once_posts_to_slack(tmp_path, monkeypatch):
    today = datetime(2026, 5, 5, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setenv("PRISM_INSTRUMENTS", "XAUUSD")
    monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PRISM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("PRISM_WATCHDOG_LOG", str(tmp_path / "watchdog.log"))
    monkeypatch.setenv("PRISM_DIGEST_LOG", str(tmp_path / "digest.log"))
    (tmp_path / "models").mkdir()

    slack = MagicMock(return_value=True)
    msg = dd.run_once(today=today, slack_fn=slack)
    slack.assert_called_once()
    assert "PRISM Daily" in msg
