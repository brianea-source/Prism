"""
PRISM Phase 4 — MT5 reconnect + heartbeat.

Locks in the behavior that keeps PRISM visible during broker outages:

* Heartbeat detects disconnect via ``terminal_info()``
* Exponential backoff between reconnect attempts (capped)
* Recovery clears disconnect state + resets alert flag
* ``supports_live_bars`` flips False while disconnected so the runner
  routes to skip (not demo) on a live deployment
* One-shot Slack alert after sustained outage (no spam)
* Credentials stashed at ``connect()`` are replayed on reinit (no env re-read)
* Runner wiring: ``_scan_instrument`` short-circuits on down link
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fake MT5 module. Swapped into MT5Bridge._mt5 directly — avoids the real
# MetaTrader5 wheel (Windows-only) and lets us toggle heartbeat results
# between scan ticks.
# ---------------------------------------------------------------------------

class _FakeTerminalInfo:
    def __init__(self, connected=True):
        self.connected = connected


class _FakeAccountInfo:
    def __init__(self, login=123, balance=10_000.0, server="Test-MT5"):
        self.login = login
        self.balance = balance
        self.server = server


class _FakeMT5:
    """Programmable MT5 stub. Set attributes per-test."""
    def __init__(self):
        self._terminal_alive = True
        self._init_next = True
        self._account_info = _FakeAccountInfo()
        self.init_calls = 0
        self.shutdown_calls = 0
        self.init_kwargs_history = []

    def initialize(self, **kwargs):
        self.init_calls += 1
        self.init_kwargs_history.append(kwargs)
        return self._init_next

    def shutdown(self):
        self.shutdown_calls += 1

    def terminal_info(self):
        if self._terminal_alive is None:
            return None
        return _FakeTerminalInfo(connected=self._terminal_alive)

    def account_info(self):
        return self._account_info

    def last_error(self):
        return (1, "fake error")


def _connected_bridge(fake_mt5=None, **kwargs):
    """Build an MT5Bridge wired to a fake MT5 module, already connected."""
    from prism.execution.mt5_bridge import MT5Bridge
    fake = fake_mt5 or _FakeMT5()
    b = MT5Bridge(mode="CONFIRM", **kwargs)
    b._mt5 = fake
    b._connected = True
    b._last_connect_kwargs = {
        "login": 123, "password": "pw", "server": "Test-MT5",
    }
    return b, fake


# ===========================================================================
# Heartbeat
# ===========================================================================

class TestHeartbeat:
    def test_alive_when_terminal_info_reports_connected(self):
        b, fake = _connected_bridge()
        assert b._heartbeat_ok() is True

    def test_dead_when_terminal_info_returns_none(self):
        b, fake = _connected_bridge()
        fake._terminal_alive = None  # terminal_info() -> None
        assert b._heartbeat_ok() is False

    def test_dead_when_terminal_connected_false(self):
        """Terminal is running but broker link is down."""
        b, fake = _connected_bridge()
        fake._terminal_alive = False
        assert b._heartbeat_ok() is False

    def test_dead_on_raise(self):
        b, fake = _connected_bridge()
        def _boom(*a, **kw): raise RuntimeError("segfault")
        fake.terminal_info = _boom
        assert b._heartbeat_ok() is False

    def test_dead_when_mt5_module_none(self):
        from prism.execution.mt5_bridge import MT5Bridge
        b = MT5Bridge()  # never connected
        assert b._heartbeat_ok() is False


# ===========================================================================
# ensure_connected: happy path
# ===========================================================================

class TestEnsureConnectedHappyPath:
    def test_alive_link_is_no_op(self):
        b, fake = _connected_bridge()
        assert b.ensure_connected() is True
        assert fake.init_calls == 0
        assert fake.shutdown_calls == 0

    def test_alive_link_keeps_disconnect_state_clear(self):
        b, fake = _connected_bridge()
        b.ensure_connected()
        assert b.disconnected_duration_sec is None
        assert b._reconnect_attempts == 0


# ===========================================================================
# ensure_connected: disconnect + recovery
# ===========================================================================

class TestDisconnectRecovery:
    def test_first_dead_call_records_disconnect_time(self):
        b, fake = _connected_bridge()
        fake._terminal_alive = False
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        assert b.ensure_connected(now) is False
        assert b._disconnect_at == now
        assert b._reconnect_attempts == 1

    def test_disconnect_flips_supports_live_bars_false(self):
        b, fake = _connected_bridge()
        assert b.supports_live_bars() is True
        fake._terminal_alive = False
        fake._init_next = False  # reinit will also fail
        b.ensure_connected(datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc))
        assert b.supports_live_bars() is False, \
            "Downstream code must see supports_live_bars=False while disconnected"

    def test_reconnect_success_clears_state(self):
        b, fake = _connected_bridge()
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        # Go down
        fake._terminal_alive = False
        fake._init_next = False
        b.ensure_connected(now)
        assert b._disconnect_at is not None

        # Come back up — next scan both heartbeat and init succeed
        fake._terminal_alive = True
        fake._init_next = True
        later = now + timedelta(seconds=15)
        # Heartbeat succeeds before we even get to the init path
        assert b.ensure_connected(later) is True
        assert b._disconnect_at is None
        assert b._reconnect_attempts == 0
        assert b._disconnect_alert_sent is False
        assert b._connected is True

    def test_reconnect_via_reinit_clears_state(self):
        """Heartbeat still dead but reinit + heartbeat-after succeed."""
        b, fake = _connected_bridge()
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        # First call: heartbeat fails, init fails → cooldown scheduled
        fake._terminal_alive = False
        fake._init_next = False
        b.ensure_connected(now)
        assert b._next_reconnect_at is not None

        # Force cooldown to have elapsed + reinit now succeeds (terminal
        # comes back after the reinit call)
        fake._init_next = True
        def _ok_after_init(**kw):
            fake.init_calls += 1
            fake.init_kwargs_history.append(kw)
            fake._terminal_alive = True  # terminal_info starts reporting alive
            return True
        fake.initialize = _ok_after_init

        later = now + timedelta(seconds=999)  # way past any cooldown
        assert b.ensure_connected(later) is True
        assert b._disconnect_at is None
        assert b._reconnect_attempts == 0


# ===========================================================================
# Exponential backoff
# ===========================================================================

class TestExponentialBackoff:
    def test_backoff_doubles_after_each_failure(self):
        b, fake = _connected_bridge(
            reconnect_base_cooldown_sec=10,
            reconnect_max_cooldown_sec=300,
        )
        fake._terminal_alive = False
        fake._init_next = False
        t0 = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        # 1st failure -> cooldown 10s
        b.ensure_connected(t0)
        assert b._next_reconnect_at == t0 + timedelta(seconds=10)

        # Within cooldown -> no new attempt
        init_calls_before = fake.init_calls
        b.ensure_connected(t0 + timedelta(seconds=5))
        assert fake.init_calls == init_calls_before, \
            "Must not reinit within cooldown window"
        assert b._reconnect_attempts == 1

        # Past cooldown -> 2nd failure, cooldown doubles to 20s
        t1 = t0 + timedelta(seconds=15)
        b.ensure_connected(t1)
        assert b._reconnect_attempts == 2
        assert b._next_reconnect_at == t1 + timedelta(seconds=20)

        # 3rd failure -> 40s
        t2 = t1 + timedelta(seconds=25)
        b.ensure_connected(t2)
        assert b._next_reconnect_at == t2 + timedelta(seconds=40)

    def test_backoff_caps_at_max(self):
        b, fake = _connected_bridge(
            reconnect_base_cooldown_sec=10,
            reconnect_max_cooldown_sec=60,  # cap early
        )
        fake._terminal_alive = False
        fake._init_next = False
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        # Fire 8 failures — without a cap, this would reach 10 * 2^7 = 1280s.
        for i in range(8):
            b.ensure_connected(now)
            now = now + timedelta(seconds=1000)

        assert b._reconnect_attempts == 8
        # Distance between last scheduled retry and its "now" is capped
        # at the max cooldown.
        assert (b._next_reconnect_at - now + timedelta(seconds=1000)).total_seconds() <= 60


# ===========================================================================
# Credentials replay — ensure_connected must reuse connect()'s kwargs
# ===========================================================================

class TestCredentialsReplay:
    def test_reinit_uses_stashed_connect_kwargs(self):
        b, fake = _connected_bridge()
        b._last_connect_kwargs = {
            "login": 7777, "password": "secret", "server": "Exness-MT5Real",
        }
        fake._terminal_alive = False
        fake._init_next = False
        b.ensure_connected(datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc))
        assert fake.init_kwargs_history, "ensure_connected must attempt reinit"
        last_kwargs = fake.init_kwargs_history[-1]
        assert last_kwargs["login"] == 7777
        assert last_kwargs["password"] == "secret"
        assert last_kwargs["server"] == "Exness-MT5Real"

    def test_shutdown_attempted_before_reinit(self):
        """Clears any half-open state before retrying initialize()."""
        b, fake = _connected_bridge()
        fake._terminal_alive = False
        fake._init_next = False
        b.ensure_connected(datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc))
        assert fake.shutdown_calls >= 1


# ===========================================================================
# One-shot disconnect Slack alert
# ===========================================================================

class TestDisconnectAlert:
    def test_should_alert_only_after_threshold(self):
        b, fake = _connected_bridge(disconnect_alert_threshold_sec=120)
        fake._terminal_alive = False
        fake._init_next = False
        t0 = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        b.ensure_connected(t0)
        # Immediately after disconnect — below threshold
        assert b.should_alert_disconnect(t0 + timedelta(seconds=30)) is False
        # Past threshold
        assert b.should_alert_disconnect(t0 + timedelta(seconds=120)) is True

    def test_alert_is_one_shot(self):
        b, fake = _connected_bridge(disconnect_alert_threshold_sec=60)
        fake._terminal_alive = False
        fake._init_next = False
        t0 = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        b.ensure_connected(t0)
        t1 = t0 + timedelta(seconds=120)
        assert b.should_alert_disconnect(t1) is True
        b.mark_disconnect_alert_sent()
        assert b.should_alert_disconnect(t1 + timedelta(seconds=60)) is False

    def test_alert_flag_resets_on_reconnect(self):
        """A new outage after recovery must be able to alert again."""
        b, fake = _connected_bridge(disconnect_alert_threshold_sec=60)
        fake._terminal_alive = False
        fake._init_next = False
        t0 = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        b.ensure_connected(t0)
        b.mark_disconnect_alert_sent()

        # Recover
        fake._terminal_alive = True
        b.ensure_connected(t0 + timedelta(seconds=200))
        assert b._disconnect_alert_sent is False

        # New outage later in the day — should alert again
        fake._terminal_alive = False
        fake._init_next = False
        t1 = t0 + timedelta(hours=3)
        b.ensure_connected(t1)
        assert b.should_alert_disconnect(t1 + timedelta(seconds=120)) is True

    def test_disconnected_duration_none_when_connected(self):
        b, _ = _connected_bridge()
        assert b.disconnected_duration_sec is None


# ===========================================================================
# Mock bridge overrides
# ===========================================================================

class TestMockBridge:
    def test_ensure_connected_always_true(self):
        from prism.execution.mt5_bridge import MockMT5Bridge
        b = MockMT5Bridge()
        assert b.ensure_connected() is True

    def test_should_alert_always_false(self):
        from prism.execution.mt5_bridge import MockMT5Bridge
        b = MockMT5Bridge()
        assert b.should_alert_disconnect() is False

    def test_disconnected_duration_is_none(self):
        from prism.execution.mt5_bridge import MockMT5Bridge
        b = MockMT5Bridge()
        assert b.disconnected_duration_sec is None


# ===========================================================================
# Runner wiring — skip scan on live disconnect (NOT fall to demo path)
# ===========================================================================

class _SignalGenSpy:
    invoked = False

    def __init__(self, *a, **kw):
        type(self).invoked = True

    def generate(self, *a, **kw):
        return None


class _StubNotifier:
    def __init__(self):
        self.alerts = []
        self.channel = "#t"
        self.client = object()
        self.confirm_timeout_sec = 300

    def send_alert(self, text):
        self.alerts.append(text)
        return "ts-alert"

    def send_signal(self, *a, **kw):
        pytest.fail("send_signal must not fire while MT5 is down")

    def update_signal_status(self, *a, **kw):
        pass


class _CleanGuard:
    def refresh(self, now): pass

    @property
    def is_tripped(self): return False

    @property
    def needs_notification(self): return False

    @property
    def snapshot(self): return {"realized_pnl_usd": 0.0}


class _DownBridge:
    """Stub bridge that reports link down — models a real MT5 in outage."""
    mode = "CONFIRM"

    def __init__(self, duration=150, alert_due=True):
        self.dur = duration
        self._alert_due = alert_due
        self.alert_sent = False
        self.ensure_calls = 0

    def ensure_connected(self, now=None):
        self.ensure_calls += 1
        return False

    def should_alert_disconnect(self, now=None):
        return self._alert_due and not self.alert_sent

    @property
    def disconnected_duration_sec(self):
        return self.dur

    def mark_disconnect_alert_sent(self):
        self.alert_sent = True

    # Not expected to be called — but keeps getattr() sweeps safe
    def supports_live_bars(self): return False
    def get_bars(self, *a, **kw): pytest.fail("get_bars must not be called")


class TestRunnerWiring:
    def setup_method(self):
        import prism.delivery.runner as runner_module
        runner_module._last_signal_key.clear()
        _SignalGenSpy.invoked = False

    def test_down_bridge_skips_scan(self, monkeypatch):
        import prism.delivery.runner as runner_module
        from prism.signal import generator as gen_module
        monkeypatch.setattr(gen_module, "SignalGenerator", _SignalGenSpy)

        bridge = _DownBridge(duration=150, alert_due=True)
        notifier = _StubNotifier()
        guard = _CleanGuard()

        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc),
            guard=guard,
        )

        assert _SignalGenSpy.invoked is False, \
            "Down bridge must short-circuit before SignalGenerator is built"
        assert bridge.ensure_calls == 1
        assert len(notifier.alerts) == 1
        assert "MT5" in notifier.alerts[0]

    def test_down_bridge_alert_is_one_shot(self, monkeypatch):
        """Multiple instruments in the same scan cycle get one alert between them."""
        import prism.delivery.runner as runner_module
        from prism.signal import generator as gen_module
        monkeypatch.setattr(gen_module, "SignalGenerator", _SignalGenSpy)

        bridge = _DownBridge(duration=150, alert_due=True)
        notifier = _StubNotifier()
        guard = _CleanGuard()
        now = datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc)

        for sym in ("EURUSD", "XAUUSD", "GBPUSD"):
            runner_module._scan_instrument(sym, notifier, bridge, now, guard=guard)

        assert len(notifier.alerts) == 1

    def test_down_bridge_no_alert_below_threshold(self, monkeypatch):
        """Short blips (< alert threshold) should skip silently."""
        import prism.delivery.runner as runner_module
        from prism.signal import generator as gen_module
        monkeypatch.setattr(gen_module, "SignalGenerator", _SignalGenSpy)

        bridge = _DownBridge(duration=30, alert_due=False)
        notifier = _StubNotifier()
        guard = _CleanGuard()

        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc),
            guard=guard,
        )
        assert len(notifier.alerts) == 0, \
            "Short disconnect must not spam Slack"
        assert _SignalGenSpy.invoked is False
