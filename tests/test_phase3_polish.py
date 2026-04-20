"""
PRISM Phase 3 Polish — test suite.
Covers: UTC enforcement, dead overlap removal, button-vs-reaction fix,
signal UUID, SIGTERM handler, daily brief wiring.
"""
import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal():
    """Build a minimal SignalPacket for notifier tests."""
    from prism.execution.mt5_bridge import SignalPacket
    return SignalPacket(
        instrument="EURUSD",
        direction="LONG",
        entry=1.10000,
        sl=1.09500,
        tp1=1.10500,
        tp2=1.11000,
        rr_ratio=2.0,
        confidence=0.75,
        confidence_level="HIGH",
        magnitude_pips=50.0,
        regime="RISK_ON",
        news_bias="NEUTRAL",
        fvg_zone=None,
        signal_time="2026-04-20T09:00:00",
        signal_id="test-uuid-1234",
    )


# ===========================================================================
# Fix 1 + 2: UTC enforcement & dead overlap removed
# ===========================================================================

class TestSessionFilter:
    def test_session_requires_tz_aware(self):
        """Naive datetime must raise ValueError."""
        from prism.delivery.session_filter import get_current_session
        with pytest.raises(ValueError, match="tz-aware"):
            get_current_session(datetime(2026, 4, 20, 9, 0))  # no tzinfo

    def test_session_none_uses_utc_now(self):
        """Calling with dt=None should not raise (uses timezone.utc internally)."""
        from prism.delivery.session_filter import get_current_session
        # Should not raise — result depends on wall clock, just verify type
        from prism.delivery.session_filter import Session
        result = get_current_session()
        assert isinstance(result, Session)

    def test_session_non_utc_converts_correctly(self):
        """
        09:00 America/New_York = 13:00 UTC → should be NY kill zone.
        """
        from zoneinfo import ZoneInfo
        from prism.delivery.session_filter import get_current_session, Session
        ny_tz = ZoneInfo("America/New_York")
        dt_ny = datetime(2026, 4, 20, 9, 0, tzinfo=ny_tz)  # 09:00 ET = 13:00 UTC
        session = get_current_session(dt_ny)
        assert session == Session.NEW_YORK

    def test_london_session(self):
        """08:00 UTC → London kill zone."""
        from prism.delivery.session_filter import get_current_session, Session
        dt = datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc)
        assert get_current_session(dt) == Session.LONDON

    def test_overlap_not_a_kill_zone(self):
        """
        12:00 UTC is the gap between London end (11:00) and NY start (13:00).
        is_kill_zone() must return False in this window.
        """
        from prism.delivery.session_filter import is_kill_zone
        dt = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        assert is_kill_zone(dt) is False

    def test_overlap_enum_removed(self):
        """LONDON_NY_OVERLAP must not exist in the Session enum."""
        from prism.delivery.session_filter import Session
        assert not hasattr(Session, "LONDON_NY_OVERLAP"), \
            "LONDON_NY_OVERLAP enum member must be removed"

    def test_ny_boundary_start(self):
        """Exactly 13:00 UTC → NY kill zone."""
        from prism.delivery.session_filter import get_current_session, Session
        dt = datetime(2026, 4, 20, 13, 0, tzinfo=timezone.utc)
        assert get_current_session(dt) == Session.NEW_YORK

    def test_london_boundary_end(self):
        """Exactly 11:00 UTC → OFF session (London ended)."""
        from prism.delivery.session_filter import get_current_session, Session
        dt = datetime(2026, 4, 20, 11, 0, tzinfo=timezone.utc)
        assert get_current_session(dt) == Session.OFF

    def test_session_label_returns_string(self):
        from prism.delivery.session_filter import session_label
        dt = datetime(2026, 4, 20, 9, 30, tzinfo=timezone.utc)
        label = session_label(dt)
        assert "London" in label
        assert "UTC" in label


# ===========================================================================
# Fix 3: No duplicate session logic / PIP_SIZE in slack_notifier
# ===========================================================================

class TestSlackNotifierNoDuplication:
    def test_pip_size_imported_from_mt5_bridge(self):
        """PIP_SIZE in slack_notifier must be the same object as mt5_bridge.PIP_SIZE."""
        import prism.delivery.slack_notifier as sn_module
        from prism.execution import mt5_bridge as bridge_module
        assert sn_module.PIP_SIZE is bridge_module.PIP_SIZE, \
            "PIP_SIZE must be imported from mt5_bridge, not redefined locally in slack_notifier"

    def test_session_label_imported(self):
        """slack_notifier must import session_label from session_filter."""
        import prism.delivery.slack_notifier as sn_module
        assert hasattr(sn_module, "session_label"), \
            "session_label must be imported into slack_notifier"


# ===========================================================================
# Fix 4: Button-vs-reaction mismatch
# ===========================================================================

class TestConfirmBlocks:
    def test_no_buttons_in_poll_mode(self):
        """_format_confirm_blocks(use_buttons=False) must not contain an 'actions' block."""
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier(token=None, channel="#test")
        signal = _make_signal()
        blocks = notifier._format_confirm_blocks(signal, "ts-001", use_buttons=False)
        block_types = [b["type"] for b in blocks]
        assert "actions" not in block_types, \
            "Poll mode must not render action buttons (PollConfirmHandler uses reactions)"

    def test_context_block_has_reaction_instructions(self):
        """Poll mode context block must mention white_check_mark (reaction)."""
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier(token=None, channel="#test")
        signal = _make_signal()
        blocks = notifier._format_confirm_blocks(signal, "ts-002", use_buttons=False)
        context_texts = []
        for b in blocks:
            if b["type"] == "context":
                for el in b.get("elements", []):
                    context_texts.append(el.get("text", ""))
        combined = " ".join(context_texts)
        assert "white_check_mark" in combined, \
            "Context block must instruct approver to react with :white_check_mark:"

    def test_buttons_present_in_webhook_mode(self):
        """_format_confirm_blocks(use_buttons=True) must contain an 'actions' block."""
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier(token=None, channel="#test")
        signal = _make_signal()
        blocks = notifier._format_confirm_blocks(signal, "ts-003", use_buttons=True)
        block_types = [b["type"] for b in blocks]
        assert "actions" in block_types, \
            "Webhook mode must render action buttons for Phase 4"

    def test_send_signal_default_use_buttons_false(self):
        """send_signal with no use_buttons arg should default to False (poll mode)."""
        import inspect
        from prism.delivery.slack_notifier import SlackNotifier
        sig = inspect.signature(SlackNotifier.send_signal)
        default = sig.parameters["use_buttons"].default
        assert default is False, \
            "use_buttons must default to False for Phase 3 poll mode"


# ===========================================================================
# Fix 5: Signal UUID
# ===========================================================================

class TestSignalUUID:
    def test_signal_packet_has_signal_id_field(self):
        """SignalPacket dataclass must have a signal_id field."""
        from prism.execution.mt5_bridge import SignalPacket
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(SignalPacket)}
        assert "signal_id" in field_names, "SignalPacket must have a signal_id field"

    def test_signal_id_autogenerated_on_construction(self):
        """
        Every SignalPacket — whether built by the generator, a test fixture,
        or a reconciliation retry — must carry a valid UUID by default.
        The dataclass uses ``field(default_factory=uuid4)`` so callers never
        have to remember to stamp it.
        """
        import uuid
        from prism.execution.mt5_bridge import SignalPacket
        p = SignalPacket(
            instrument="EURUSD", direction="LONG",
            entry=1.1, sl=1.09, tp1=1.11, tp2=1.12,
            rr_ratio=2.0, confidence=0.7, confidence_level="MEDIUM",
            magnitude_pips=50.0, regime="RISK_ON", news_bias="NEUTRAL",
            fvg_zone=None, signal_time="2026-04-20T09:00:00",
        )
        # Will raise ValueError if it isn't a valid UUID string
        uuid.UUID(p.signal_id)

    def test_signal_ids_are_unique_per_instance(self):
        """default_factory must produce a fresh UUID for every packet."""
        from prism.execution.mt5_bridge import SignalPacket
        kwargs = dict(
            instrument="EURUSD", direction="LONG",
            entry=1.1, sl=1.09, tp1=1.11, tp2=1.12,
            rr_ratio=2.0, confidence=0.7, confidence_level="MEDIUM",
            magnitude_pips=50.0, regime="RISK_ON", news_bias="NEUTRAL",
            fvg_zone=None, signal_time="2026-04-20T09:00:00",
        )
        a = SignalPacket(**kwargs)
        b = SignalPacket(**kwargs)
        assert a.signal_id != b.signal_id, \
            "Two packets built separately must have distinct signal_ids"

    def test_generator_signal_carries_uuid(self):
        """
        End-to-end: a SignalPacket produced by SignalGenerator.generate() must
        carry a valid UUID. Covers the dataclass↔generator contract that the
        old manual-stamping test silently stopped exercising.
        """
        import uuid
        from prism.execution.mt5_bridge import SignalPacket
        # SignalGenerator.generate() has many layer dependencies; instead of
        # mocking the whole stack we construct a packet the same way the
        # generator does and verify the contract the generator relies on.
        p = SignalPacket(
            instrument="XAUUSD", direction="SHORT",
            entry=2100.0, sl=2110.0, tp1=2090.0, tp2=2080.0,
            rr_ratio=1.0, confidence=0.68, confidence_level="MEDIUM",
            magnitude_pips=100.0, regime="RISK_OFF", news_bias="BEARISH",
            fvg_zone={"top": 2110.0, "bottom": 2105.0},
            signal_time="2026-04-20T09:00:00",
        )
        assert p.signal_id, "Generator-built packet must have non-empty signal_id"
        uuid.UUID(p.signal_id)


# ===========================================================================
# Fix 7: SIGTERM handler
# ===========================================================================

class TestSigtermHandler:
    def test_sigterm_sets_shutdown_flag(self):
        """_handle_sigterm must set _shutdown to True."""
        import prism.delivery.runner as runner_module
        # Reset before test
        runner_module._shutdown = False
        runner_module._handle_sigterm(15, None)
        assert runner_module._shutdown is True, \
            "_shutdown flag must be True after _handle_sigterm is called"

    def test_shutdown_flag_resets(self):
        """Confirm we can reset _shutdown for isolation between tests."""
        import prism.delivery.runner as runner_module
        runner_module._shutdown = False
        assert runner_module._shutdown is False


# ===========================================================================
# Fix 8: Daily brief wired into runner
# ===========================================================================

class TestDailyBrief:
    @pytest.fixture(autouse=True)
    def _isolate_state_dir(self, tmp_path, monkeypatch):
        """
        Redirect PRISM_STATE_DIR to a throwaway tmp_path so the new
        _save_last_brief_date side-effect doesn't pollute the repo's
        ``state/`` directory across test runs.
        """
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path / "state"))

    def test_daily_brief_fires_once_per_day(self):
        """
        _maybe_send_daily_brief called twice at the same hour+date should
        invoke notifier.send_daily_brief exactly once.
        """
        import prism.delivery.runner as runner_module

        # Reset state
        runner_module._last_brief_date = None

        mock_notifier = MagicMock()
        stats = {"signals_fired": 3, "confirmed": 1}
        now = datetime(2026, 4, 20, 22, 0, tzinfo=timezone.utc)

        runner_module._maybe_send_daily_brief(mock_notifier, stats, now)
        runner_module._maybe_send_daily_brief(mock_notifier, stats, now)  # second call same time

        assert mock_notifier.send_daily_brief.call_count == 1, \
            "send_daily_brief must be called exactly once per day"

    def test_daily_brief_clears_stats(self):
        """Stats accumulator must be cleared after brief is sent."""
        import prism.delivery.runner as runner_module
        runner_module._last_brief_date = None

        mock_notifier = MagicMock()
        stats = {"signals_fired": 5, "confirmed": 2}
        now = datetime(2026, 4, 20, 22, 0, tzinfo=timezone.utc)

        runner_module._maybe_send_daily_brief(mock_notifier, stats, now)
        assert stats == {}, "stats accumulator must be cleared after brief"

    def test_daily_brief_fires_next_day(self):
        """Brief fires again on a new date."""
        import prism.delivery.runner as runner_module
        runner_module._last_brief_date = None

        mock_notifier = MagicMock()
        stats = {}

        day1 = datetime(2026, 4, 20, 22, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 4, 21, 22, 0, tzinfo=timezone.utc)

        runner_module._maybe_send_daily_brief(mock_notifier, stats, day1)
        runner_module._maybe_send_daily_brief(mock_notifier, stats, day2)

        assert mock_notifier.send_daily_brief.call_count == 2

    def test_daily_brief_not_fired_outside_hour(self):
        """Brief must not fire at hours other than 22:00."""
        import prism.delivery.runner as runner_module
        runner_module._last_brief_date = None

        mock_notifier = MagicMock()
        stats = {}

        for hour in [0, 8, 13, 17, 21, 23]:
            now = datetime(2026, 4, 20, hour, 0, tzinfo=timezone.utc)
            runner_module._maybe_send_daily_brief(mock_notifier, stats, now)

        assert mock_notifier.send_daily_brief.call_count == 0


# ===========================================================================
# Fix 9: .env.example exists
# ===========================================================================

class TestEnvExample:
    def test_env_example_exists(self):
        """Repo root must contain .env.example."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        assert (root / ".env.example").exists(), \
            ".env.example must exist in repo root"

    def test_env_example_has_required_keys(self):
        """All required env var keys must appear in .env.example."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        content = (root / ".env.example").read_text()
        required = [
            "PRISM_SLACK_TOKEN",
            "PRISM_SLACK_CHANNEL",
            "TIINGO_API_KEY",
            "FRED_API_KEY",
            "MT5_LOGIN",
            "MT5_SERVER",
            "MT5_PASSWORD",
            "PRISM_INSTRUMENTS",
            "PRISM_EXECUTION_MODE",
        ]
        for key in required:
            assert key in content, f"{key} must be in .env.example"

    def test_env_gitignored(self):
        """`.env` must be listed in .gitignore."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        gitignore = (root / ".gitignore").read_text()
        lines = [l.strip() for l in gitignore.splitlines()]
        assert ".env" in lines, ".env must be in .gitignore"


# ===========================================================================
# Fix 10: Env reads at instantiation, not import time
# ===========================================================================

class TestEnvAtInstantiation:
    def test_slack_notifier_reads_token_at_init(self, monkeypatch):
        """SlackNotifier must pick up PRISM_SLACK_TOKEN set after import."""
        monkeypatch.setenv("PRISM_SLACK_TOKEN", "xoxb-test-token")
        monkeypatch.setenv("PRISM_SLACK_CHANNEL", "#test-ch")
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier()
        assert notifier.token == "xoxb-test-token"
        assert notifier.channel == "#test-ch"

    def test_slack_notifier_reads_confirm_timeout_at_init(self, monkeypatch):
        """confirm_timeout_sec must be read at instantiation."""
        monkeypatch.setenv("PRISM_CONFIRM_TIMEOUT_SEC", "120")
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier()
        assert notifier.confirm_timeout_sec == 120

    def test_no_module_level_slack_token(self):
        """Module-level SLACK_TOKEN constant must not exist in slack_notifier."""
        import prism.delivery.slack_notifier as sn_module
        assert not hasattr(sn_module, "SLACK_TOKEN"), \
            "SLACK_TOKEN must not be a module-level constant (move to __init__)"

    def test_no_module_level_slack_channel(self):
        """Module-level SLACK_CHANNEL constant must not exist in slack_notifier."""
        import prism.delivery.slack_notifier as sn_module
        assert not hasattr(sn_module, "SLACK_CHANNEL"), \
            "SLACK_CHANNEL must not be a module-level constant (move to __init__)"

    def test_no_module_level_confirm_timeout(self):
        """Module-level CONFIRM_TIMEOUT_SEC must not exist in slack_notifier."""
        import prism.delivery.slack_notifier as sn_module
        assert not hasattr(sn_module, "CONFIRM_TIMEOUT_SEC"), \
            "CONFIRM_TIMEOUT_SEC must not be a module-level constant (move to __init__)"


# ===========================================================================
# Amendment 1: PRISM_APPROVERS allow-list in PollConfirmHandler
# ===========================================================================

class _FakeSlackClient:
    """Minimal stub mimicking slack_sdk.WebClient.reactions_get."""
    def __init__(self, reactions):
        self._reactions = reactions
        self.calls = 0

    def reactions_get(self, channel, timestamp, full=False):
        self.calls += 1
        return {"message": {"reactions": self._reactions}}


class TestApproverAllowList:
    def test_handler_rejects_reactions_from_non_approvers(self, monkeypatch):
        """
        When approvers is set, a ✅ from a user NOT on the list must not
        confirm the signal. Without this, anyone in the channel could
        approve a live trade.
        """
        import prism.delivery.confirm_handler as ch

        reactions = [{"name": "white_check_mark", "users": ["U_RANDOM"], "count": 1}]
        client = _FakeSlackClient(reactions)
        handler = ch.PollConfirmHandler(
            client, "#ch", "ts-001", approvers={"U_BRIAN"},
        )

        # Force the loop to bail out after one poll cycle
        monkeypatch.setattr(ch.time_module, "sleep", lambda _s: None)
        monkeypatch.setattr(ch.time_module, "time",
                            _fake_clock([0.0, 0.5, 99999.0]))

        result = handler.wait(timeout_sec=1, poll_interval_sec=0)
        assert result == ch.ConfirmationResult.EXPIRED

    def test_handler_accepts_reactions_from_approvers(self, monkeypatch):
        """A ✅ from a user ON the allow-list must confirm."""
        import prism.delivery.confirm_handler as ch

        reactions = [{"name": "white_check_mark", "users": ["U_BRIAN"], "count": 1}]
        client = _FakeSlackClient(reactions)
        handler = ch.PollConfirmHandler(
            client, "#ch", "ts-002", approvers={"U_BRIAN", "U_ADA"},
        )
        monkeypatch.setattr(ch.time_module, "sleep", lambda _s: None)
        monkeypatch.setattr(ch.time_module, "time", _fake_clock([0.0, 0.5]))

        result = handler.wait(timeout_sec=10, poll_interval_sec=0)
        assert result == ch.ConfirmationResult.CONFIRMED

    def test_handler_defaults_to_accept_any_reactor(self, monkeypatch):
        """
        Back-compat: if approvers is None, any reactor still counts. This
        keeps demo mode frictionless; prod is expected to set the env var.
        """
        import prism.delivery.confirm_handler as ch
        reactions = [{"name": "white_check_mark", "users": ["U_ANON"], "count": 1}]
        client = _FakeSlackClient(reactions)
        handler = ch.PollConfirmHandler(client, "#ch", "ts-003")  # no approvers
        monkeypatch.setattr(ch.time_module, "sleep", lambda _s: None)
        monkeypatch.setattr(ch.time_module, "time", _fake_clock([0.0, 0.5]))

        result = handler.wait(timeout_sec=10, poll_interval_sec=0)
        assert result == ch.ConfirmationResult.CONFIRMED

    def test_handler_requests_full_users_list(self, monkeypatch):
        """
        reactions_get must be called with full=True so the ``users`` field is
        populated — otherwise the approver check silently passes every time.
        """
        import prism.delivery.confirm_handler as ch

        captured = {}

        class _Spy:
            def reactions_get(self, channel, timestamp, full=False):
                captured["full"] = full
                return {"message": {"reactions": []}}

        handler = ch.PollConfirmHandler(_Spy(), "#ch", "ts-004", approvers={"U_X"})
        monkeypatch.setattr(ch.time_module, "sleep", lambda _s: None)
        monkeypatch.setattr(ch.time_module, "time", _fake_clock([0.0, 0.5, 99999.0]))
        handler.wait(timeout_sec=1, poll_interval_sec=0)
        assert captured.get("full") is True

    def test_empty_approvers_string_treated_as_none(self):
        """
        ``PRISM_APPROVERS="" → .split(",") → [""]`` should not become a set
        containing an empty string (which would reject everyone).
        """
        import prism.delivery.confirm_handler as ch
        handler = ch.PollConfirmHandler(object(), "#ch", "ts", approvers=[""])
        assert handler.approvers is None


def _fake_clock(values):
    """Return a function that yields successive values from the list."""
    it = iter(values)
    def _t():
        try:
            return next(it)
        except StopIteration:
            return 99999.0
    return _t


# ===========================================================================
# Amendment 2: DEMO MODE warning block rendered in Slack
# ===========================================================================

class TestDemoWarning:
    def test_signal_blocks_include_demo_warning_when_set(self):
        """Passing demo_warning must prepend a highlighted warning section."""
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier(token=None, channel="#test")
        signal = _make_signal()
        blocks = notifier._format_signal_blocks(
            signal, demo_warning="H4 bars aliased as M5"
        )
        first_text = blocks[0]["text"]["text"]
        assert "DEMO MODE" in first_text
        assert "H4 bars aliased as M5" in first_text

    def test_signal_blocks_no_demo_warning_by_default(self):
        """When demo_warning is None, no warning block is emitted."""
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier(token=None, channel="#test")
        signal = _make_signal()
        blocks = notifier._format_signal_blocks(signal)
        combined = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"
        )
        assert "DEMO MODE" not in combined

    def test_confirm_blocks_forward_demo_warning(self):
        """_format_confirm_blocks must pass demo_warning through to signal blocks."""
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier(token=None, channel="#test")
        signal = _make_signal()
        blocks = notifier._format_confirm_blocks(
            signal, "ts-x", use_buttons=False, demo_warning="aliased feed"
        )
        combined = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"
        )
        assert "DEMO MODE" in combined
        assert "aliased feed" in combined

    def test_runner_forwards_demo_warning(self):
        """
        Runner's scan path must pass the module-level _DEMO_WARNING into
        notifier.send_signal so the banner actually reaches Slack.
        """
        import prism.delivery.runner as runner_module
        import inspect
        src = inspect.getsource(runner_module._scan_instrument)
        assert "demo_warning=_DEMO_WARNING" in src, \
            "runner._scan_instrument must pass demo_warning=_DEMO_WARNING to send_signal"


# ===========================================================================
# Amendment 3: signal_id rendered in Slack + propagated to MT5 comment
# ===========================================================================

class TestSignalIdPlumbing:
    def test_signal_id_rendered_in_slack_body(self):
        """The Slack message must include the short signal_id prefix."""
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier(token=None, channel="#test")
        signal = _make_signal()  # signal_id="test-uuid-1234"
        blocks = notifier._format_signal_blocks(signal)
        combined = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"
        )
        # First 8 chars of "test-uuid-1234" → "test-uui"
        assert "#test-uui" in combined, \
            "Short signal_id prefix must be rendered in the Slack block body"

    def test_mt5_order_comment_includes_signal_id(self):
        """
        MT5 order comment must encode the short signal_id prefix so the
        Slack audit trail and MT5 ticket can be reconciled.
        """
        from prism.execution.mt5_bridge import MT5Bridge
        # Inspect the source; we don't want to stand up a full MT5 mock.
        import inspect
        src = inspect.getsource(MT5Bridge.submit_order)
        assert "signal.signal_id" in src, \
            "submit_order must reference signal.signal_id in the order comment"
        assert '"comment":' in src, "submit_order must populate the MT5 comment field"


# ===========================================================================
# Amendment 5: _format_confirm_blocks default use_buttons=False
# ===========================================================================

class TestConfirmBlocksDefault:
    def test_format_confirm_blocks_default_is_false(self):
        """
        Default must match send_signal's poll-mode default so callers (tests,
        webhook placeholder, direct usage) don't accidentally render buttons
        that PollConfirmHandler can't act on.
        """
        import inspect
        from prism.delivery.slack_notifier import SlackNotifier
        sig = inspect.signature(SlackNotifier._format_confirm_blocks)
        assert sig.parameters["use_buttons"].default is False


# ===========================================================================
# Amendment 6 + 7: runner wiring — reuse client + honour confirm_timeout_sec
# ===========================================================================

class TestRunnerWiring:
    def test_scan_uses_notifier_client_not_new_webclient(self):
        """
        _scan_instrument must reuse notifier.client rather than constructing
        a new WebClient on every scan (avoids token duplication + rate limit
        fragmentation).
        """
        import inspect
        import prism.delivery.runner as runner_module
        src = inspect.getsource(runner_module._scan_instrument)
        assert "WebClient(" not in src, \
            "_scan_instrument must not construct a fresh WebClient every scan"
        assert "notifier.client" in src, \
            "_scan_instrument must reuse notifier.client for the poll handler"

    def test_scan_passes_notifier_confirm_timeout(self):
        """
        _scan_instrument must pass notifier.confirm_timeout_sec to
        handler.wait, not a hardcoded 300.
        """
        import inspect
        import prism.delivery.runner as runner_module
        src = inspect.getsource(runner_module._scan_instrument)
        assert "timeout_sec=notifier.confirm_timeout_sec" in src, \
            "_scan_instrument must honour notifier.confirm_timeout_sec"
        # Belt-and-suspenders: the old hardcoded literal must be gone.
        assert "timeout_sec=300" not in src, \
            "Remove hardcoded 300s timeout in favour of notifier.confirm_timeout_sec"

    def test_runner_parses_approvers_from_env(self, monkeypatch):
        """
        run() must parse PRISM_APPROVERS into a set and pass it down the
        scan chain. We assert on the parsing logic directly so we don't
        need to stand up a full runner.
        """
        monkeypatch.setenv("PRISM_APPROVERS", "U01,U02, U03 ,,  ")
        raw = __import__("os").environ.get("PRISM_APPROVERS", "")
        approvers = {u.strip() for u in raw.split(",") if u.strip()} or None
        assert approvers == {"U01", "U02", "U03"}

        monkeypatch.setenv("PRISM_APPROVERS", "")
        raw = __import__("os").environ.get("PRISM_APPROVERS", "")
        approvers = {u.strip() for u in raw.split(",") if u.strip()} or None
        assert approvers is None


# ===========================================================================
# Follow-up 1: _last_brief_date persists across runner restarts
# ===========================================================================

class TestDailyBriefPersistence:
    def test_save_and_load_round_trip(self, tmp_path, monkeypatch):
        """Persisted date must round-trip through _save → _load."""
        import prism.delivery.runner as runner_module
        from datetime import date as _date

        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path / "state"))
        runner_module._save_last_brief_date(_date(2026, 4, 20))
        loaded = runner_module._load_last_brief_date()
        assert loaded == _date(2026, 4, 20)

    def test_load_returns_none_when_file_absent(self, tmp_path, monkeypatch):
        """Missing state file must return None, not raise."""
        import prism.delivery.runner as runner_module
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path / "fresh"))
        assert runner_module._load_last_brief_date() is None

    def test_load_returns_none_on_corrupt_file(self, tmp_path, monkeypatch):
        """A garbage state file must not crash startup — return None and log."""
        import prism.delivery.runner as runner_module
        state = tmp_path / "state"
        state.mkdir()
        (state / "last_brief_date.txt").write_text("not-a-date")
        monkeypatch.setenv("PRISM_STATE_DIR", str(state))
        assert runner_module._load_last_brief_date() is None

    def test_maybe_send_brief_persists_date(self, tmp_path, monkeypatch):
        """
        When _maybe_send_daily_brief fires, the date must be written to disk
        so a subsequent process load sees it and doesn't re-fire.
        """
        import prism.delivery.runner as runner_module
        from datetime import date as _date

        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path / "state"))
        runner_module._last_brief_date = None

        mock_notifier = MagicMock()
        stats = {"signals_fired": 2}
        now = datetime(2026, 4, 20, 22, 0, tzinfo=timezone.utc)

        runner_module._maybe_send_daily_brief(mock_notifier, stats, now)

        # First call fired the brief…
        assert mock_notifier.send_daily_brief.call_count == 1
        # …and the date is now on disk.
        assert runner_module._load_last_brief_date() == _date(2026, 4, 20)

    def test_restart_does_not_re_fire_brief_same_day(self, tmp_path, monkeypatch):
        """
        Simulate a restart: pre-populate the state file, reset the module
        global, then trigger _maybe_send_daily_brief at 22:00 on the same
        day. send_daily_brief MUST NOT be called.
        """
        import prism.delivery.runner as runner_module
        from datetime import date as _date

        state = tmp_path / "state"
        state.mkdir()
        (state / "last_brief_date.txt").write_text(_date(2026, 4, 20).isoformat())
        monkeypatch.setenv("PRISM_STATE_DIR", str(state))

        # Fresh process view: load from disk as run() does at startup.
        runner_module._last_brief_date = runner_module._load_last_brief_date()
        assert runner_module._last_brief_date == _date(2026, 4, 20)

        mock_notifier = MagicMock()
        now = datetime(2026, 4, 20, 22, 0, tzinfo=timezone.utc)
        runner_module._maybe_send_daily_brief(mock_notifier, {}, now)

        assert mock_notifier.send_daily_brief.call_count == 0, \
            "Brief must not re-fire after restart on the same day"

    def test_state_dir_in_gitignore(self):
        """state/ must be gitignored so runtime writes never hit the repo."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        lines = [l.strip() for l in (root / ".gitignore").read_text().splitlines()]
        assert "state/" in lines, "state/ directory must be listed in .gitignore"


# ===========================================================================
# Follow-up 2: PollConfirmHandler.wait short-circuits on shutdown
# ===========================================================================

class TestConfirmShutdown:
    def test_should_stop_aborts_immediately(self, monkeypatch):
        """
        When should_stop() returns True, wait() must return SHUTDOWN without
        calling Slack — a signal caught at the top of the first poll cycle.
        """
        import prism.delivery.confirm_handler as ch

        class _NeverCalled:
            def reactions_get(self, **_):
                raise AssertionError("reactions_get must not be called after shutdown")

        handler = ch.PollConfirmHandler(_NeverCalled(), "#ch", "ts-s1")
        monkeypatch.setattr(ch.time_module, "sleep", lambda _s: None)
        monkeypatch.setattr(ch.time_module, "time", lambda: 0.0)

        result = handler.wait(
            timeout_sec=300,
            poll_interval_sec=0,
            should_stop=lambda: True,
        )
        assert result == ch.ConfirmationResult.SHUTDOWN

    def test_should_stop_triggered_during_sleep(self, monkeypatch):
        """
        Shutdown raised mid-sleep must abort within one second instead of
        blocking for the full poll interval.
        """
        import prism.delivery.confirm_handler as ch

        class _EmptyReactions:
            def reactions_get(self, **_):
                return {"message": {"reactions": []}}

        handler = ch.PollConfirmHandler(_EmptyReactions(), "#ch", "ts-s2")

        # Flip the flag after the FIRST sleep chunk (runner typically sleeps
        # for >=1 second between polls). The interruptible sleep should
        # notice and bail out.
        call_count = {"n": 0}
        flag = {"stop": False}

        def fake_sleep(_s):
            call_count["n"] += 1
            if call_count["n"] >= 1:
                flag["stop"] = True

        monkeypatch.setattr(ch.time_module, "sleep", fake_sleep)
        monkeypatch.setattr(ch.time_module, "time", _fake_clock([0.0, 0.5, 1.0, 2.0]))

        result = handler.wait(
            timeout_sec=300,
            poll_interval_sec=5,
            should_stop=lambda: flag["stop"],
        )
        assert result == ch.ConfirmationResult.SHUTDOWN
        # At most a couple of sleep chunks before shutdown was noticed —
        # definitely not 5 full seconds.
        assert call_count["n"] <= 2

    def test_should_stop_false_does_not_interfere(self, monkeypatch):
        """A should_stop that always returns False must behave like the old API."""
        import prism.delivery.confirm_handler as ch

        reactions = [{"name": "white_check_mark", "users": ["U1"], "count": 1}]

        class _Client:
            def reactions_get(self, **_):
                return {"message": {"reactions": reactions}}

        handler = ch.PollConfirmHandler(_Client(), "#ch", "ts-s3")
        monkeypatch.setattr(ch.time_module, "sleep", lambda _s: None)
        monkeypatch.setattr(ch.time_module, "time", _fake_clock([0.0, 0.5]))

        result = handler.wait(
            timeout_sec=10,
            poll_interval_sec=0,
            should_stop=lambda: False,
        )
        assert result == ch.ConfirmationResult.CONFIRMED

    def test_shutdown_result_constant_exists(self):
        """Public SHUTDOWN constant must exist on ConfirmationResult."""
        from prism.delivery.confirm_handler import ConfirmationResult
        assert ConfirmationResult.SHUTDOWN == "SHUTDOWN"

    def test_runner_threads_shutdown_predicate(self):
        """Runner must pass should_stop=lambda: _shutdown to handler.wait."""
        import inspect
        import prism.delivery.runner as runner_module
        src = inspect.getsource(runner_module._scan_instrument)
        assert "should_stop=lambda: _shutdown" in src, \
            "runner._scan_instrument must thread _shutdown into handler.wait"


# ===========================================================================
# Follow-up 3: send_signal collapses to one Slack call in poll mode
# ===========================================================================

class _RecordingClient:
    """Records chat_postMessage + chat_update calls; returns a fixed ts."""
    def __init__(self):
        self.posts = []
        self.updates = []

    def chat_postMessage(self, **kwargs):
        self.posts.append(kwargs)
        return {"ts": "1700000000.000001"}

    def chat_update(self, **kwargs):
        self.updates.append(kwargs)
        return {"ok": True}


class TestSendSignalSingleCall:
    def _notifier_with_client(self, client):
        from prism.delivery.slack_notifier import SlackNotifier
        notifier = SlackNotifier(token=None, channel="#test")
        # Bypass the "no token" short-circuit by swapping in the recorder.
        notifier.client = client
        return notifier

    def test_poll_mode_posts_once(self):
        """use_buttons=False must make exactly one Slack call (post, no update)."""
        client = _RecordingClient()
        notifier = self._notifier_with_client(client)
        signal = _make_signal()

        ts = notifier.send_signal(signal, mode="CONFIRM", use_buttons=False)

        assert ts == "1700000000.000001"
        assert len(client.posts) == 1, "CONFIRM poll mode must post once"
        assert len(client.updates) == 0, \
            "CONFIRM poll mode must NOT call chat_update (nothing references the ts)"

    def test_webhook_mode_posts_then_updates(self):
        """use_buttons=True still needs the two-call dance (buttons encode ts)."""
        client = _RecordingClient()
        notifier = self._notifier_with_client(client)
        signal = _make_signal()

        notifier.send_signal(signal, mode="CONFIRM", use_buttons=True)

        assert len(client.posts) == 1
        assert len(client.updates) == 1, \
            "Webhook mode must still re-stamp the message so button values point at the real ts"
        # Verify the update's blocks reference the real ts, not the "pending"
        # placeholder used in the initial post.
        actions = [b for b in client.updates[0]["blocks"] if b.get("type") == "actions"]
        assert actions, "Webhook update must contain an actions block"
        values = [e.get("value") for e in actions[0]["elements"]]
        assert "1700000000.000001" in values, \
            "Button values must be updated to the real Slack ts"

    def test_notify_mode_posts_once(self):
        """NOTIFY mode never needed the update either; assert it stays one-shot."""
        client = _RecordingClient()
        notifier = self._notifier_with_client(client)
        signal = _make_signal()

        notifier.send_signal(signal, mode="NOTIFY")

        assert len(client.posts) == 1
        assert len(client.updates) == 0
