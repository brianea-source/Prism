"""
PRISM Phase 3 tests — session filter, Slack notifier, confirmation handler.
All tests are self-contained and require no live Slack token or MT5 connection.
"""
import sys
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Minimal SignalPacket stand-in for tests (avoids importing full prism stack)
# ---------------------------------------------------------------------------
@dataclass
class MockSignalPacket:
    instrument: str = "XAUUSD"
    direction: str = "LONG"
    entry: float = 2385.50
    sl: float = 2371.20
    tp1: float = 2400.50
    tp2: float = 2428.40
    rr_ratio: float = 1.0
    confidence: float = 0.74
    confidence_level: str = "MEDIUM"
    magnitude_pips: float = 150.0
    regime: str = "RISK_OFF"
    news_bias: str = "NEUTRAL"
    fvg_zone: Optional[dict] = None
    signal_time: str = "2026-04-20T08:30:00"
    model_version: str = "prism_v2.0"


# ---------------------------------------------------------------------------
# Session filter tests
# ---------------------------------------------------------------------------
class TestSessionFilter:
    def _dt(self, hour: int, minute: int = 0) -> datetime:
        """Build a UTC datetime for a given hour."""
        return datetime(2026, 4, 20, hour, minute, 0, tzinfo=timezone.utc)

    def test_session_london_kill_zone(self):
        from prism.delivery.session_filter import is_kill_zone
        assert is_kill_zone(self._dt(9, 0)) is True

    def test_session_ny_kill_zone(self):
        from prism.delivery.session_filter import is_kill_zone
        assert is_kill_zone(self._dt(14, 30)) is True

    def test_session_asian_not_kill_zone(self):
        from prism.delivery.session_filter import is_kill_zone
        assert is_kill_zone(self._dt(3, 0)) is False

    def test_session_off_not_kill_zone(self):
        """11:30 UTC is between London close and NY open — not a kill zone."""
        from prism.delivery.session_filter import is_kill_zone
        assert is_kill_zone(self._dt(11, 30)) is False

    def test_london_boundary_open(self):
        from prism.delivery.session_filter import is_kill_zone
        assert is_kill_zone(self._dt(7, 0)) is True

    def test_london_boundary_close(self):
        """11:00 UTC is exactly London close — should NOT be kill zone."""
        from prism.delivery.session_filter import is_kill_zone
        assert is_kill_zone(self._dt(11, 0)) is False

    def test_ny_boundary_open(self):
        from prism.delivery.session_filter import is_kill_zone
        assert is_kill_zone(self._dt(13, 0)) is True

    def test_ny_boundary_close(self):
        """17:00 UTC is exactly NY close — should NOT be kill zone."""
        from prism.delivery.session_filter import is_kill_zone
        assert is_kill_zone(self._dt(17, 0)) is False

    def test_london_ny_overlap(self):
        """No overlap window exists for XAUUSD (London closes at 11, NY opens at 13)."""
        from prism.delivery.session_filter import get_current_session, Session
        # 13:00-15:00 overlap only applies if LONDON_END > NY_START — it doesn't,
        # so this should return NEW_YORK, not LONDON_NY_OVERLAP.
        session = get_current_session(self._dt(14, 0))
        assert session == Session.NEW_YORK

    def test_session_label_format(self):
        from prism.delivery.session_filter import session_label
        label = session_label(self._dt(9, 30))
        assert "UTC" in label
        assert "09:30" in label

    def test_session_label_off_session(self):
        from prism.delivery.session_filter import session_label
        label = session_label(self._dt(12, 0))
        assert "Off-session" in label
        assert "UTC" in label


# ---------------------------------------------------------------------------
# Slack notifier tests (no real API calls)
# ---------------------------------------------------------------------------
class TestSlackNotifier:
    """All Slack tests use no token to avoid real API calls."""

    def _make_notifier(self, token: str = ""):
        from prism.delivery.slack_notifier import SlackNotifier
        return SlackNotifier(token=token, channel="#test-signals")

    def test_slack_notifier_formats_signal(self):
        notifier = self._make_notifier()
        signal = MockSignalPacket()

        # Patch SignalPacket import: SlackNotifier accepts anything duck-typed
        blocks = notifier._format_signal_blocks(signal)  # type: ignore[arg-type]

        assert isinstance(blocks, list)
        assert len(blocks) >= 1

        # Find the section block with mrkdwn text
        section = next(
            (b for b in blocks if b.get("type") == "section"),
            None,
        )
        assert section is not None, "Expected a 'section' block"
        text = section["text"]["text"]
        assert "LONG" in text
        assert "Entry" in text

    def test_slack_notifier_formats_fvg_zone(self):
        notifier = self._make_notifier()
        signal = MockSignalPacket(
            fvg_zone={"timeframe": "H4", "bottom": 2378.40, "top": 2382.60, "partially_mitigated": True}
        )
        blocks = notifier._format_signal_blocks(signal)  # type: ignore[arg-type]
        section = next(b for b in blocks if b.get("type") == "section")
        assert "2378.40" in section["text"]["text"]
        assert "2382.60" in section["text"]["text"]

    def test_slack_notifier_no_token_returns_none(self):
        notifier = self._make_notifier(token="")
        signal = MockSignalPacket()
        result = notifier.send_signal(signal, mode="NOTIFY")  # type: ignore[arg-type]
        assert result is None

    def test_slack_notifier_client_none_when_no_token(self):
        notifier = self._make_notifier(token="")
        assert notifier.client is None

    def test_slack_notifier_update_status_no_token_no_crash(self):
        """update_signal_status must not raise when client is None."""
        notifier = self._make_notifier(token="")
        signal = MockSignalPacket()
        # Should return silently
        notifier.update_signal_status("fake_ts", "EXECUTED", signal)  # type: ignore[arg-type]

    def test_slack_notifier_confirm_blocks_has_actions(self):
        notifier = self._make_notifier()
        signal = MockSignalPacket()
        blocks = notifier._format_confirm_blocks(signal, "pending")  # type: ignore[arg-type]
        types = [b["type"] for b in blocks]
        assert "actions" in types

    def test_slack_notifier_confirm_blocks_has_context_timeout(self):
        notifier = self._make_notifier()
        signal = MockSignalPacket()
        blocks = notifier._format_confirm_blocks(signal, "pending")  # type: ignore[arg-type]
        context_blocks = [b for b in blocks if b["type"] == "context"]
        assert context_blocks, "Expected a context block with timeout info"
        text = context_blocks[0]["elements"][0]["text"]
        assert "minute" in text

    def test_slack_notifier_calc_rr(self):
        notifier = self._make_notifier()
        rr = notifier._calc_rr(entry=2385.50, sl=2371.20, tp=2400.50)
        assert rr > 0
        assert isinstance(rr, float)

    def test_slack_notifier_calc_rr_zero_sl_distance(self):
        notifier = self._make_notifier()
        rr = notifier._calc_rr(entry=2385.50, sl=2385.50, tp=2400.50)
        assert rr == 0.0


# ---------------------------------------------------------------------------
# Confirmation handler tests
# ---------------------------------------------------------------------------
class TestConfirmHandler:
    def test_confirm_handler_parse_webhook_confirm(self):
        from prism.delivery.confirm_handler import WebhookConfirmHandler
        payload = {"actions": [{"action_id": "prism_confirm", "value": "12345"}]}
        action_id, value = WebhookConfirmHandler.parse_interaction(payload)
        assert action_id == "prism_confirm"
        assert value == "12345"

    def test_confirm_handler_parse_webhook_skip(self):
        from prism.delivery.confirm_handler import WebhookConfirmHandler
        payload = {"actions": [{"action_id": "prism_skip", "value": "99999"}]}
        action_id, value = WebhookConfirmHandler.parse_interaction(payload)
        assert action_id == "prism_skip"
        assert value == "99999"

    def test_confirm_handler_parse_empty_actions(self):
        from prism.delivery.confirm_handler import WebhookConfirmHandler
        action_id, value = WebhookConfirmHandler.parse_interaction({"actions": []})
        assert action_id == ""
        assert value == ""

    def test_confirm_handler_parse_missing_actions_key(self):
        from prism.delivery.confirm_handler import WebhookConfirmHandler
        action_id, value = WebhookConfirmHandler.parse_interaction({})
        assert action_id == ""
        assert value == ""

    def test_confirmation_result_constants(self):
        from prism.delivery.confirm_handler import ConfirmationResult
        assert ConfirmationResult.CONFIRMED == "CONFIRMED"
        assert ConfirmationResult.SKIPPED == "SKIPPED"
        assert ConfirmationResult.EXPIRED == "EXPIRED"

    def test_poll_handler_expires_immediately(self):
        """PollConfirmHandler with a stub client and 0-second timeout returns EXPIRED."""
        from prism.delivery.confirm_handler import PollConfirmHandler, ConfirmationResult

        class StubClient:
            def reactions_get(self, **kwargs):
                return {"message": {"reactions": []}}

        handler = PollConfirmHandler(
            client=StubClient(),
            channel="#test",
            message_ts="12345.6789",
        )
        result = handler.wait(timeout_sec=0, poll_interval_sec=0)
        assert result == ConfirmationResult.EXPIRED

    def test_poll_handler_detects_confirm_reaction(self):
        from prism.delivery.confirm_handler import PollConfirmHandler, ConfirmationResult

        class StubClient:
            def reactions_get(self, **kwargs):
                return {"message": {"reactions": [{"name": "white_check_mark", "count": 1}]}}

        handler = PollConfirmHandler(StubClient(), "#test", "12345.6789")
        result = handler.wait(timeout_sec=5, poll_interval_sec=0)
        assert result == ConfirmationResult.CONFIRMED

    def test_poll_handler_detects_skip_reaction(self):
        from prism.delivery.confirm_handler import PollConfirmHandler, ConfirmationResult

        class StubClient:
            def reactions_get(self, **kwargs):
                return {"message": {"reactions": [{"name": "x", "count": 1}]}}

        handler = PollConfirmHandler(StubClient(), "#test", "12345.6789")
        result = handler.wait(timeout_sec=5, poll_interval_sec=0)
        assert result == ConfirmationResult.SKIPPED
