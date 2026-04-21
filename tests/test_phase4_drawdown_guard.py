"""
PRISM Phase 4 — daily drawdown kill-switch.

Locks in the behaviour that keeps a bad news day from becoming an account-
killing day:

* Trip conditions (pct of SOD balance + absolute USD cap, OR semantics)
* Auto-reset at UTC midnight (state file carries across day boundaries)
* One-shot Slack notification (no spam while halted)
* Persistence: a mid-day restart does NOT clear the counter
* MT5 deal-history sync is idempotent on re-observation (no double-count)
* Wiring into runner._scan_instrument pre-empts signal generation
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Stub bridges. Real MT5Bridge depends on MetaTrader5 (Windows-only wheel);
# the guard only calls get_account_balance + deals_since_utc_midnight so
# we stub those directly.
# ---------------------------------------------------------------------------

class _StubBridge:
    """Minimal bridge stub the guard needs: balance + deal history."""
    mode = "NOTIFY"

    def __init__(self, balance=10_000.0, deals=None):
        self._balance = balance
        self._deals = deals or []
        self.balance_calls = 0
        self.deals_calls = 0

    def get_account_balance(self):
        self.balance_calls += 1
        return self._balance

    def deals_since_utc_midnight(self, now=None, magic_number=None):
        self.deals_calls += 1
        return list(self._deals)  # copy — guard shouldn't mutate

    def supports_live_bars(self):
        return False

    # Phase-4 live-bars surface — tests that exercise the demo path need
    # a valid DataFrame so _scan_instrument can reach the generator.
    def get_bars(self, instrument, timeframe, count=500):
        dates = pd.date_range(
            "2026-04-18", periods=60, freq="4h", tz="UTC",
        )
        return pd.DataFrame({
            "datetime": dates,
            "open": 1.10, "high": 1.11, "low": 1.09,
            "close": 1.10, "volume": 100,
        })

    def bars_are_fresh(self, df, tf, now=None):
        return True


# ===========================================================================
# Trip thresholds
# ===========================================================================

class TestTripThresholds:
    def test_pct_threshold_trips_at_3pct_loss(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        bridge = _StubBridge(balance=10_000.0)
        g = DrawdownGuard(bridge, tmp_path, max_daily_loss_pct=0.03)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        g.refresh(now)
        assert g.is_tripped is False

        # $299 loss = still under the 3% ($300) threshold
        g.record_manual(-299.0, now=now)
        assert g.is_tripped is False

        # $301 loss = over threshold
        g.record_manual(-2.0, now=now)
        assert g.is_tripped is True

    def test_absolute_usd_threshold_trips(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        bridge = _StubBridge(balance=100_000.0)  # pct threshold = $3000
        # Absolute cap $200 is MUCH tighter
        g = DrawdownGuard(bridge, tmp_path,
                          max_daily_loss_pct=0.03,
                          max_daily_loss_usd=200.0)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        g.refresh(now)
        g.record_manual(-150.0, now=now)
        assert g.is_tripped is False

        g.record_manual(-60.0, now=now)  # cumulative = -$210
        assert g.is_tripped is True

    def test_more_restrictive_cap_wins(self, tmp_path):
        """When BOTH caps are set, the tighter one fires first."""
        from prism.delivery.drawdown_guard import DrawdownGuard
        bridge = _StubBridge(balance=10_000.0)  # pct cap = -$300
        # Absolute cap -$500 is LESS restrictive on a $10k account
        g = DrawdownGuard(bridge, tmp_path,
                          max_daily_loss_pct=0.03,
                          max_daily_loss_usd=500.0)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        g.refresh(now)
        # $350 loss trips the pct cap but is below the $500 abs cap
        g.record_manual(-350.0, now=now)
        assert g.is_tripped is True

    def test_profit_never_trips(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        g = DrawdownGuard(_StubBridge(10_000.0), tmp_path)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        g.refresh(now)
        g.record_manual(+500.0, now=now)
        g.record_manual(+200.0, now=now)
        assert g.is_tripped is False


# ===========================================================================
# UTC midnight reset
# ===========================================================================

class TestMidnightReset:
    def test_new_utc_day_resets_state(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        g = DrawdownGuard(_StubBridge(10_000.0), tmp_path,
                          max_daily_loss_pct=0.03)
        day1 = datetime(2026, 4, 21, 14, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 4, 22, 0, 5, tzinfo=timezone.utc)  # 5 min past UTC midnight

        g.refresh(day1)
        g.record_manual(-400.0, now=day1)
        assert g.is_tripped is True

        # New UTC day = state resets
        g.refresh(day2)
        assert g.is_tripped is False
        assert g.snapshot["realized_pnl_usd"] == 0.0
        assert g.snapshot["date"] == "2026-04-22"

    def test_same_day_refresh_preserves_state(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        g = DrawdownGuard(_StubBridge(10_000.0), tmp_path)
        t1 = datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 4, 21, 18, 0, tzinfo=timezone.utc)

        g.refresh(t1)
        g.record_manual(-100.0, now=t1)
        g.refresh(t2)  # Same UTC day
        assert g.snapshot["realized_pnl_usd"] == pytest.approx(-100.0)

    def test_refresh_requires_tz_aware_datetime(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        g = DrawdownGuard(_StubBridge(10_000.0), tmp_path)
        with pytest.raises(ValueError, match="tz-aware"):
            g.refresh(datetime(2026, 4, 21, 8, 0))  # naive


# ===========================================================================
# Persistence — crucial because a restart during a halt must NOT re-enable
# trading. The state file carries the trip across processes.
# ===========================================================================

class TestPersistence:
    def test_trip_survives_process_restart(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        bridge = _StubBridge(10_000.0)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        g1 = DrawdownGuard(bridge, tmp_path, max_daily_loss_pct=0.03)
        g1.refresh(now)
        g1.record_manual(-400.0, now=now)
        assert g1.is_tripped is True

        # New process — fresh object, same state dir
        g2 = DrawdownGuard(bridge, tmp_path, max_daily_loss_pct=0.03)
        assert g2.is_tripped is True, \
            "Trip state must survive process restart (halted day should stay halted)"

    def test_notified_flag_persists(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        bridge = _StubBridge(10_000.0)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        g1 = DrawdownGuard(bridge, tmp_path, max_daily_loss_pct=0.03)
        g1.refresh(now)
        g1.record_manual(-400.0, now=now)
        g1.mark_notified()

        g2 = DrawdownGuard(bridge, tmp_path, max_daily_loss_pct=0.03)
        assert g2.is_tripped is True
        assert g2.needs_notification is False, \
            "Restart must NOT re-fire the Slack trip alert"

    def test_corrupt_state_file_starts_fresh(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        (tmp_path / "daily_drawdown.json").write_text("{not valid json")
        g = DrawdownGuard(_StubBridge(10_000.0), tmp_path)
        assert g.is_tripped is False

    def test_missing_state_dir_is_created(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        sub = tmp_path / "does" / "not" / "exist" / "yet"
        g = DrawdownGuard(_StubBridge(10_000.0), sub)
        assert sub.exists()


# ===========================================================================
# MT5 deal-history sync
# ===========================================================================

class TestDealSync:
    def test_deals_add_to_realized_pnl(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        bridge = _StubBridge(10_000.0, deals=[
            {"ticket": 1, "profit": -50.0, "symbol": "EURUSD", "time": 0},
            {"ticket": 2, "profit": -30.0, "symbol": "XAUUSD", "time": 0},
        ])
        g = DrawdownGuard(bridge, tmp_path)
        g.refresh(datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc))
        assert g.snapshot["realized_pnl_usd"] == pytest.approx(-80.0)

    def test_deals_are_idempotent_on_repeat_refresh(self, tmp_path):
        """
        Scanner calls refresh() every 60s. Deals already counted on the
        previous call must not be counted again — otherwise a single $50
        losing trade becomes $50 * scans_per_day in cumulative PnL.
        """
        from prism.delivery.drawdown_guard import DrawdownGuard
        deals = [{"ticket": 1, "profit": -50.0, "symbol": "EURUSD", "time": 0}]
        bridge = _StubBridge(10_000.0, deals=deals)
        g = DrawdownGuard(bridge, tmp_path)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        g.refresh(now)
        g.refresh(now)  # scanner tick #2
        g.refresh(now)  # scanner tick #3
        assert g.snapshot["realized_pnl_usd"] == pytest.approx(-50.0)

    def test_new_deals_arrive_and_are_added(self, tmp_path):
        """After first sync, new deals should be picked up."""
        from prism.delivery.drawdown_guard import DrawdownGuard
        bridge = _StubBridge(10_000.0, deals=[
            {"ticket": 1, "profit": -50.0, "symbol": "EURUSD", "time": 0},
        ])
        g = DrawdownGuard(bridge, tmp_path)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        g.refresh(now)
        assert g.snapshot["realized_pnl_usd"] == pytest.approx(-50.0)

        # New deal arrives
        bridge._deals.append({"ticket": 2, "profit": -20.0, "symbol": "EURUSD", "time": 1})
        g.refresh(now)
        assert g.snapshot["realized_pnl_usd"] == pytest.approx(-70.0)

    def test_deal_sync_failure_is_survivable(self, tmp_path):
        """A broker hiccup on history_deals_get must not crash the guard."""
        from prism.delivery.drawdown_guard import DrawdownGuard

        class _FlakyBridge(_StubBridge):
            def deals_since_utc_midnight(self, now=None, magic_number=None):
                raise RuntimeError("mt5 history timeout")

        g = DrawdownGuard(_FlakyBridge(10_000.0), tmp_path)
        # Should not raise — guard logs and carries on with prior state
        g.refresh(datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc))
        assert g.is_tripped is False


# ===========================================================================
# One-shot Slack notification
# ===========================================================================

class TestOneShotNotification:
    def test_needs_notification_true_when_tripped_and_not_notified(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        g = DrawdownGuard(_StubBridge(10_000.0), tmp_path,
                          max_daily_loss_pct=0.03)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        g.refresh(now)
        g.record_manual(-400.0, now=now)
        assert g.needs_notification is True

    def test_mark_notified_clears_needs_flag(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        g = DrawdownGuard(_StubBridge(10_000.0), tmp_path,
                          max_daily_loss_pct=0.03)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        g.refresh(now)
        g.record_manual(-400.0, now=now)
        g.mark_notified()
        assert g.is_tripped is True
        assert g.needs_notification is False

    def test_format_alert_contains_key_info(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        g = DrawdownGuard(_StubBridge(10_000.0), tmp_path,
                          max_daily_loss_pct=0.03)
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        g.refresh(now)
        g.record_manual(-400.0, now=now)
        alert = g.format_alert()
        assert "halted" in alert.lower()
        # Dollar figure must appear so Brian sees the magnitude at a glance
        assert "400" in alert
        # Threshold pct must appear so he can see why
        assert "3.0%" in alert


# ===========================================================================
# Factory / env parsing
# ===========================================================================

class TestBuildGuardFromEnv:
    def test_defaults_to_3pct(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRISM_MAX_DAILY_LOSS_PCT", raising=False)
        monkeypatch.delenv("PRISM_MAX_DAILY_LOSS_USD", raising=False)
        from prism.delivery.drawdown_guard import build_guard_from_env
        g = build_guard_from_env(_StubBridge(10_000.0), tmp_path)
        assert g.max_daily_loss_pct == pytest.approx(0.03)
        assert g.max_daily_loss_usd is None

    def test_env_overrides_pct_and_usd(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_MAX_DAILY_LOSS_PCT", "0.05")
        monkeypatch.setenv("PRISM_MAX_DAILY_LOSS_USD", "250")
        from prism.delivery.drawdown_guard import build_guard_from_env
        g = build_guard_from_env(_StubBridge(10_000.0), tmp_path)
        assert g.max_daily_loss_pct == pytest.approx(0.05)
        assert g.max_daily_loss_usd == pytest.approx(250.0)

    def test_empty_usd_string_parses_to_none(self, tmp_path, monkeypatch):
        """Blank PRISM_MAX_DAILY_LOSS_USD must disable the abs cap, not crash."""
        monkeypatch.setenv("PRISM_MAX_DAILY_LOSS_USD", "")
        from prism.delivery.drawdown_guard import build_guard_from_env
        g = build_guard_from_env(_StubBridge(10_000.0), tmp_path)
        assert g.max_daily_loss_usd is None


# ===========================================================================
# Runner wiring — guard pre-empts signal generation
# ===========================================================================

class _SignalGenSpy:
    """Tracks whether SignalGenerator was instantiated."""
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
        pytest.fail("send_signal must not be called while guard is tripped")

    def update_signal_status(self, *a, **kw):
        pass


class _TrippedGuard:
    """Guard stub that reports as tripped immediately."""
    def __init__(self):
        self.refresh_calls = 0
        self.notify_calls = 0
        self._notified = False

    def refresh(self, now):
        self.refresh_calls += 1

    @property
    def is_tripped(self): return True

    @property
    def needs_notification(self): return not self._notified

    def mark_notified(self):
        self._notified = True
        self.notify_calls += 1

    def format_alert(self):
        return ":octagonal_sign: halted"

    @property
    def snapshot(self):
        return {"realized_pnl_usd": -400.0}


class _CleanGuard:
    """Guard stub that reports not tripped (happy path)."""
    def refresh(self, now): pass

    @property
    def is_tripped(self): return False

    @property
    def needs_notification(self): return False

    @property
    def snapshot(self):
        return {"realized_pnl_usd": 0.0}


class TestRunnerWiring:
    def setup_method(self):
        import prism.delivery.runner as runner_module
        runner_module._last_signal_key.clear()
        _SignalGenSpy.invoked = False

    def test_tripped_guard_skips_signal_generation(self, monkeypatch):
        import prism.delivery.runner as runner_module
        from prism.signal import generator as gen_module

        monkeypatch.setattr(gen_module, "SignalGenerator", _SignalGenSpy)

        bridge = _StubBridge(10_000.0)
        notifier = _StubNotifier()
        guard = _TrippedGuard()

        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc),
            guard=guard,
        )

        assert _SignalGenSpy.invoked is False, \
            "Tripped guard must short-circuit before SignalGenerator is constructed"
        assert guard.refresh_calls == 1
        assert len(notifier.alerts) == 1
        assert "halted" in notifier.alerts[0].lower()

    def test_tripped_notify_is_one_shot(self, monkeypatch):
        import prism.delivery.runner as runner_module
        from prism.signal import generator as gen_module

        monkeypatch.setattr(gen_module, "SignalGenerator", _SignalGenSpy)

        bridge = _StubBridge(10_000.0)
        notifier = _StubNotifier()
        guard = _TrippedGuard()
        now = datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc)

        runner_module._scan_instrument("EURUSD", notifier, bridge, now, guard=guard)
        runner_module._scan_instrument("XAUUSD", notifier, bridge, now, guard=guard)
        runner_module._scan_instrument("GBPUSD", notifier, bridge, now, guard=guard)

        assert len(notifier.alerts) == 1, \
            "Multiple scans with tripped guard must still only post ONE Slack alert"

    def test_clean_guard_does_not_skip(self, monkeypatch):
        """Sanity: a non-tripped guard lets the scan proceed normally."""
        import prism.delivery.runner as runner_module
        from prism.signal import generator as gen_module

        # Stub SignalGenerator to return None — we only need to prove the
        # scan path was entered (not short-circuited by the guard).
        monkeypatch.setattr(gen_module, "SignalGenerator", _SignalGenSpy)

        bridge = _StubBridge(10_000.0)
        notifier = _StubNotifier()
        guard = _CleanGuard()

        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc),
            guard=guard,
        )
        assert _SignalGenSpy.invoked is True, \
            "Clean guard must let scan proceed past the guard gate"


# ===========================================================================
# MT5Bridge.deals_since_utc_midnight
# ===========================================================================

class _FakeDeal:
    def __init__(self, ticket, profit, magic, symbol="EURUSD", time=0):
        self.ticket = ticket
        self.profit = profit
        self.magic = magic
        self.symbol = symbol
        self.time = time


class _FakeMt5:
    def __init__(self, deals=None, raise_on_get=False):
        self._deals = deals or []
        self._raise = raise_on_get
        self.calls = []

    def history_deals_get(self, dt_from, dt_to):
        self.calls.append((dt_from, dt_to))
        if self._raise:
            raise RuntimeError("boom")
        return self._deals


class TestDealsSinceUtcMidnight:
    def _bridge(self, fake):
        from prism.execution.mt5_bridge import MT5Bridge
        b = MT5Bridge(mode="CONFIRM")
        b._mt5 = fake
        b._connected = True
        return b

    def test_filters_by_magic(self):
        from prism.execution.mt5_bridge import MAGIC_NUMBER
        fake = _FakeMt5(deals=[
            _FakeDeal(1, -50.0, MAGIC_NUMBER),
            _FakeDeal(2, -20.0, 999),  # not a PRISM deal
            _FakeDeal(3, +30.0, MAGIC_NUMBER),
        ])
        bridge = self._bridge(fake)
        out = bridge.deals_since_utc_midnight(
            now=datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc),
        )
        tickets = [d["ticket"] for d in out]
        assert 1 in tickets and 3 in tickets
        assert 2 not in tickets

    def test_window_starts_at_utc_midnight(self):
        from prism.execution.mt5_bridge import MAGIC_NUMBER
        fake = _FakeMt5(deals=[_FakeDeal(1, -10.0, MAGIC_NUMBER)])
        bridge = self._bridge(fake)
        now = datetime(2026, 4, 21, 14, 23, tzinfo=timezone.utc)
        bridge.deals_since_utc_midnight(now=now)
        dt_from, dt_to = fake.calls[0]
        assert dt_from == datetime(2026, 4, 21, 0, 0, tzinfo=timezone.utc)
        assert dt_to == now

    def test_disconnected_returns_empty(self):
        from prism.execution.mt5_bridge import MT5Bridge
        b = MT5Bridge(mode="CONFIRM")  # not connected
        assert b.deals_since_utc_midnight() == []

    def test_mt5_error_returns_empty(self):
        bridge = self._bridge(_FakeMt5(raise_on_get=True))
        assert bridge.deals_since_utc_midnight(
            now=datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc),
        ) == []

    def test_mock_bridge_override_returns_empty(self):
        from prism.execution.mt5_bridge import MockMT5Bridge
        b = MockMT5Bridge()
        b.connect()
        assert b.deals_since_utc_midnight() == []
