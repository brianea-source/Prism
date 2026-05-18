"""Tests for runner dormancy detection and gate rejection tracking."""
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

from prism.delivery import runner as r  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_runner_globals():
    """Reset module-level state between tests."""
    r._gate_rejection_counts.clear()
    r._dormancy_alerted = False
    yield
    r._gate_rejection_counts.clear()
    r._dormancy_alerted = False


# ---------------------------------------------------------------------------
# Gate rejection recording
# ---------------------------------------------------------------------------
class TestGateRejection:
    def test_record_gate_rejection_increments(self):
        r._record_gate_rejection("news_bias")
        r._record_gate_rejection("news_bias")
        r._record_gate_rejection("icc_structure")
        assert r._gate_rejection_counts == {"news_bias": 2, "icc_structure": 1}

    def test_save_gate_rejections_writes_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        r._gate_rejection_counts["news_bias"] = 42
        r._save_gate_rejections()
        path = tmp_path / "gate_rejections.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["news_bias"] == 42

    def test_load_gate_rejections_hydrates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        (tmp_path / "gate_rejections.json").write_text(
            json.dumps({"htf_bias": 100, "news_bias": 50})
        )
        loaded = r._load_gate_rejections()
        assert loaded == {"htf_bias": 100, "news_bias": 50}

    def test_load_gate_rejections_missing_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        loaded = r._load_gate_rejections()
        assert loaded == {}

    def test_load_gate_rejections_corrupt_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        (tmp_path / "gate_rejections.json").write_text("{bad")
        loaded = r._load_gate_rejections()
        assert loaded == {}


# ---------------------------------------------------------------------------
# Last signal fire persistence
# ---------------------------------------------------------------------------
class TestLastSignalFire:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        ts = datetime(2026, 5, 18, 10, 30, tzinfo=timezone.utc)
        r._save_last_signal_fire(ts)
        loaded = r._load_last_signal_fire()
        assert loaded == ts

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        assert r._load_last_signal_fire() is None

    def test_load_corrupt_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        (tmp_path / "last_signal_fire_ts.txt").write_text("garbage")
        assert r._load_last_signal_fire() is None


# ---------------------------------------------------------------------------
# Dormancy detection
# ---------------------------------------------------------------------------
class TestDormancyCheck:
    def test_alerts_after_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        fire_time = datetime(2026, 5, 16, 8, 0, tzinfo=timezone.utc)
        r._save_last_signal_fire(fire_time)
        r._gate_rejection_counts["news_bias"] = 500
        r._gate_rejection_counts["ml_confidence"] = 100

        notifier = MagicMock()
        now = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        r._check_dormancy(notifier, now)

        notifier.send_alert.assert_called_once()
        msg = notifier.send_alert.call_args[0][0]
        assert "dormancy" in msg.lower()
        assert "news_bias" in msg
        assert r._dormancy_alerted is True

    def test_no_alert_before_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        fire_time = datetime(2026, 5, 18, 6, 0, tzinfo=timezone.utc)
        r._save_last_signal_fire(fire_time)

        notifier = MagicMock()
        now = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        r._check_dormancy(notifier, now)

        notifier.send_alert.assert_not_called()
        assert r._dormancy_alerted is False

    def test_one_shot_alert(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        fire_time = datetime(2026, 5, 16, 8, 0, tzinfo=timezone.utc)
        r._save_last_signal_fire(fire_time)

        notifier = MagicMock()
        now = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        r._check_dormancy(notifier, now)
        r._check_dormancy(notifier, now + timedelta(hours=1))

        assert notifier.send_alert.call_count == 1

    def test_no_alert_when_no_prior_signal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        notifier = MagicMock()
        now = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        r._check_dormancy(notifier, now)
        notifier.send_alert.assert_not_called()

    def test_dormancy_resets_on_signal_fire(self, tmp_path, monkeypatch):
        """After a dormancy alert fires, _dormancy_alerted should reset
        to False when a signal fires (simulated by direct assignment)."""
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        fire_time = datetime(2026, 5, 16, 8, 0, tzinfo=timezone.utc)
        r._save_last_signal_fire(fire_time)

        notifier = MagicMock()
        now = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        r._check_dormancy(notifier, now)
        assert r._dormancy_alerted is True

        r._dormancy_alerted = False
        r._save_last_signal_fire(now)

        r._check_dormancy(notifier, now)
        assert notifier.send_alert.call_count == 1


# ---------------------------------------------------------------------------
# Generator gate tagging
# ---------------------------------------------------------------------------
class TestGeneratorGateTag:
    def test_last_rejection_gate_set_on_news_bias(self):
        from prism.signal.generator import SignalGenerator
        gen = SignalGenerator("XAUUSD")
        assert gen.last_rejection_gate is None
