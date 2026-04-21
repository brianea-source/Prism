"""
PRISM Phase 4 — correctness improvements (non-prod-breaking fixes).

Covers:
  Fix 2  — FileNotFoundError mid-scan sends Slack alert (not silently swallowed)
  Fix 3  — PRISM_BAR_COUNT env var replaces hardcoded count=500
  Fix 4  — Approximate lot sizing warning in Slack signal card
  Fix 5  — EURUSD/GBPUSD approx pip value fallback contract
  Fix 6  — guard.refresh() once per scan cycle (not once per instrument)
  Fix 7  — SOD balance snapshot logs "mid-day start" when runner starts after midnight
  Fix 8  — _heartbeat_ok() falls back to account_info() when terminal_info() fails
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from prism.execution.mt5_bridge import (
    APPROX_PIP_VALUE_PER_LOT,
    MT5Bridge,
    MockMT5Bridge,
    SignalPacket,
)
from prism.delivery.slack_notifier import SlackNotifier


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_signal(**overrides) -> SignalPacket:
    defaults = dict(
        instrument="EURUSD", direction="LONG",
        entry=1.1000, sl=1.0950, tp1=1.1050, tp2=1.1100,
        rr_ratio=1.0, confidence=0.75, confidence_level="HIGH",
        magnitude_pips=50.0, regime="RISK_ON", news_bias="NEUTRAL",
        fvg_zone=None, signal_time="2026-04-20T14:00:00",
    )
    defaults.update(overrides)
    return SignalPacket(**defaults)


def _dummy_bars(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "datetime": idx,
        "open": np.ones(n),
        "high": np.ones(n) + 0.01,
        "low": np.ones(n) - 0.01,
        "close": np.ones(n),
        "volume": np.ones(n) * 100,
    })


# ---------------------------------------------------------------------------
# Fix 3 — PRISM_BAR_COUNT env var
# ---------------------------------------------------------------------------

class TestBarCountEnvVar:
    """BAR_COUNT module constant is read from env; every get_bars() call uses it."""

    def test_default_is_500(self):
        import prism.delivery.runner as runner
        expected = int(os.environ.get("PRISM_BAR_COUNT", "500"))
        assert runner.BAR_COUNT == expected

    def test_get_bars_uses_bar_count(self, monkeypatch):
        """Patch BAR_COUNT=200; verify bridge.get_bars is called with count=200."""
        import prism.delivery.runner as runner
        monkeypatch.setattr(runner, "BAR_COUNT", 200)

        bar_counts_seen: list = []

        class _TrackedBridge(MockMT5Bridge):
            def supports_live_bars(self): return False
            def get_bars(self, instrument, timeframe, count=500):
                bar_counts_seen.append(count)
                return pd.DataFrame()  # empty → cache-miss path → early return

        notifier = MagicMock()
        bridge = _TrackedBridge()
        bridge.connect()
        now = datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc)
        runner._scan_instrument("EURUSD", notifier, bridge, now)

        assert len(bar_counts_seen) > 0, "get_bars was never called"
        for c in bar_counts_seen:
            assert c == 200, f"Expected count=200, got count={c}"


# ---------------------------------------------------------------------------
# Fix 2 — FileNotFoundError mid-scan sends Slack alert
# ---------------------------------------------------------------------------

class TestFileNotFoundHandling:
    """FileNotFoundError from SignalGenerator must alert Slack and not re-raise."""

    def _bars_bridge(self):
        dummy = _dummy_bars()
        class _BB(MockMT5Bridge):
            def supports_live_bars(self): return False
            def get_bars(self, instrument, timeframe, count=500):
                return dummy.copy()
        b = _BB(); b.connect(); return b

    def test_file_not_found_sends_alert(self, monkeypatch):
        import prism.delivery.runner as runner
        exc = FileNotFoundError("models/xgb_EURUSD.pkl")
        exc.filename = "models/xgb_EURUSD.pkl"

        class _BrokenSG:
            def __init__(self, *a, **kw): pass
            def generate(self, *a, **kw): raise exc

        monkeypatch.setattr("prism.signal.generator.SignalGenerator", _BrokenSG)

        alerts: list = []
        class _Notifier:
            def send_alert(self, text): alerts.append(text); return "ts"

        runner._scan_instrument("EURUSD", _Notifier(), self._bars_bridge(),
                                datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc))

        assert len(alerts) == 1
        assert "EURUSD" in alerts[0]
        # Either "model file missing" or similar phrasing
        assert any(kw in alerts[0].lower() for kw in ("model", "missing", "file"))

    def test_file_not_found_does_not_reraise(self, monkeypatch):
        import prism.delivery.runner as runner
        exc = FileNotFoundError("models/lgbm.pkl")
        exc.filename = "models/lgbm.pkl"

        class _BrokenSG:
            def __init__(self, *a, **kw): pass
            def generate(self, *a, **kw): raise exc

        monkeypatch.setattr("prism.signal.generator.SignalGenerator", _BrokenSG)
        notifier = MagicMock()
        # Must not raise
        runner._scan_instrument("XAUUSD", notifier, self._bars_bridge(),
                                datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc))
        notifier.send_alert.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 4 — Approximate lot sizing warning in Slack signal card
# ---------------------------------------------------------------------------

class TestApproximateSizingWarning:
    """approximate_sizing=True prepends a context block; False → no warning."""

    def _notifier(self): return SlackNotifier(token="xoxb-fake", channel="#test")

    def test_warning_block_present_when_approximate(self):
        signal = _make_signal()
        signal.approximate_sizing = True
        blocks = self._notifier()._format_signal_blocks(signal)

        assert blocks
        first = blocks[0]
        assert first.get("type") == "context", (
            f"First block should be context warning, got {first.get('type')!r}")
        text = " ".join(e.get("text", "") for e in first.get("elements", []))
        assert "pproximate" in text  # "Approximate" or "approximate"

    def test_no_warning_block_when_not_approximate(self):
        signal = _make_signal()
        signal.approximate_sizing = False
        blocks = self._notifier()._format_signal_blocks(signal)
        for block in blocks:
            if block.get("type") == "context":
                text = " ".join(e.get("text", "") for e in block.get("elements", []))
                assert "pproximate" not in text, f"Unexpected approx warning: {text!r}"


# ---------------------------------------------------------------------------
# Fix 5 — EURUSD/GBPUSD approx pip value fallback contract
# ---------------------------------------------------------------------------

class TestApproxFallback:
    """MockMT5Bridge._pip_value_per_lot: FX majors use __DEFAULT__, XAUUSD uses gold."""

    def test_eurusd_uses_default(self):
        b = MockMT5Bridge(); b.connect()
        val, source = b._pip_value_per_lot("EURUSD", "EURUSD")
        assert source == "approximation"
        assert val == APPROX_PIP_VALUE_PER_LOT["__DEFAULT__"]

    def test_gbpusd_uses_default(self):
        b = MockMT5Bridge(); b.connect()
        val, source = b._pip_value_per_lot("GBPUSD", "GBPUSD")
        assert source == "approximation"
        assert val == APPROX_PIP_VALUE_PER_LOT["__DEFAULT__"]

    def test_xauusd_uses_gold_approx(self):
        b = MockMT5Bridge(); b.connect()
        val, source = b._pip_value_per_lot("XAUUSD", "XAUUSD")
        assert source == "approximation"
        assert val == APPROX_PIP_VALUE_PER_LOT["XAUUSD"]


# ---------------------------------------------------------------------------
# Fix 6 — guard.refresh() once per scan cycle
# ---------------------------------------------------------------------------

class TestGuardRefreshOncePerCycle:
    """
    _scan_instrument must NOT call guard.refresh (Fix 6 moved it to run() loop).
    is_tripped is checked once per instrument call.
    """

    def _tripped_guard(self):
        class _TG:
            refresh_calls = []
            is_tripped_checks = 0
            def refresh(self, now): self.refresh_calls.append(now)
            @property
            def is_tripped(self):
                self.is_tripped_checks += 1
                return True
            @property
            def needs_notification(self): return False
            @property
            def snapshot(self): return {"realized_pnl_usd": 0.0}
        return _TG()

    def test_scan_instrument_does_not_refresh_guard(self):
        import prism.delivery.runner as runner
        guard = self._tripped_guard()
        runner._scan_instrument(
            "EURUSD", MagicMock(), MockMT5Bridge(),
            datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc), guard=guard,
        )
        assert len(guard.refresh_calls) == 0, (
            f"guard.refresh must not be called inside _scan_instrument; "
            f"was called {len(guard.refresh_calls)} time(s)")

    def test_is_tripped_checked_once_per_instrument(self):
        import prism.delivery.runner as runner
        instruments = ["XAUUSD", "EURUSD", "GBPUSD"]
        checks = [0]

        class _CountGuard:
            refresh_calls = []
            def refresh(self, now): self.refresh_calls.append(now)
            @property
            def is_tripped(self):
                checks[0] += 1
                return True
            @property
            def needs_notification(self): return False
            @property
            def snapshot(self): return {"realized_pnl_usd": 0.0}

        guard = _CountGuard()
        now = datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc)
        for instr in instruments:
            runner._scan_instrument(instr, MagicMock(), MockMT5Bridge(), now, guard=guard)

        assert checks[0] == len(instruments), (
            f"is_tripped should be checked once per instrument call "
            f"(expected {len(instruments)}, got {checks[0]})")
        assert len(guard.refresh_calls) == 0


# ---------------------------------------------------------------------------
# Fix 7 — SOD balance snapshot logs "mid-day start"
# ---------------------------------------------------------------------------

class TestMidDaySnapshotLog:
    """DrawdownGuard logs 'mid-day start' when refresh() is called after midnight."""

    def _guard(self, tmp_path):
        from prism.delivery.drawdown_guard import DrawdownGuard
        class _MB:
            def get_account_balance(self): return 10_000.0
            def deals_since_utc_midnight(self, now=None, magic_number=None): return []
        return DrawdownGuard(bridge=_MB(), state_dir=tmp_path, max_daily_loss_pct=0.03)

    def test_midday_start_logs_warning(self, tmp_path, caplog):
        guard = self._guard(tmp_path)
        with caplog.at_level(logging.INFO, logger="prism.delivery.drawdown_guard"):
            guard.refresh(datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc))
        msgs = [r.message for r in caplog.records if "mid-day" in r.message.lower()]
        assert msgs, f"Expected mid-day log; got: {[r.message for r in caplog.records]}"

    def test_midnight_start_no_midday_log(self, tmp_path, caplog):
        guard = self._guard(tmp_path)
        with caplog.at_level(logging.INFO, logger="prism.delivery.drawdown_guard"):
            guard.refresh(datetime(2026, 4, 20, 0, 1, tzinfo=timezone.utc))
        msgs = [r.message for r in caplog.records if "mid-day" in r.message.lower()]
        assert not msgs, f"No mid-day log expected at 00:01 UTC; got: {msgs}"


# ---------------------------------------------------------------------------
# Fix 8 — _heartbeat_ok() fallback to account_info()
# ---------------------------------------------------------------------------

class TestHeartbeatFallback:
    """_heartbeat_ok() falls back from terminal_info() to account_info()."""

    def _bridge(self, ti_result, ai_result) -> MT5Bridge:
        class _FakeMt5:
            def __init__(self, ti, ai): self._ti, self._ai = ti, ai
            def terminal_info(self): return self._ti() if callable(self._ti) else self._ti
            def account_info(self): return self._ai() if callable(self._ai) else self._ai
        b = MT5Bridge(mode="CONFIRM")
        b._connected = True
        b._mt5 = _FakeMt5(ti_result, ai_result)
        return b

    def test_terminal_info_none_falls_back_to_account_info(self):
        """terminal_info()=None → use account_info(); if it returns object → True."""
        class _FakeAcct: login = 1
        bridge = self._bridge(ti_result=None, ai_result=_FakeAcct())
        assert bridge._heartbeat_ok() is True

    def test_terminal_disconnected_is_trusted(self):
        """terminal_info().connected=False is treated as authoritative — returns False.

        Note: account_info() is NOT probed when terminal_info() explicitly
        says disconnected. This preserves the reconnect loop semantics: a
        definitive False from terminal_info engages reconnect immediately
        rather than masking the disconnect behind a broker-side check.
        (Contrast with terminal_info() returning None, which does fall back.)
        """
        class _DiscoTerm: connected = False
        class _FakeAcct: login = 2
        bridge = self._bridge(ti_result=_DiscoTerm(), ai_result=_FakeAcct())
        assert bridge._heartbeat_ok() is False

    def test_both_fail_returns_false(self):
        """terminal_info() raises AND account_info() raises → False."""
        def _raise(): raise RuntimeError("MT5 dead")
        bridge = self._bridge(ti_result=_raise, ai_result=_raise)
        assert bridge._heartbeat_ok() is False

    def test_account_info_none_returns_false(self):
        """terminal_info()=None and account_info()=None → False."""
        bridge = self._bridge(ti_result=None, ai_result=None)
        assert bridge._heartbeat_ok() is False

    def test_terminal_info_connected_true_does_not_call_account_info(self):
        """Fast path: terminal_info().connected=True → no account_info() call."""
        ai_calls = [0]
        class _ConnTerm: connected = True
        def _ai(): ai_calls[0] += 1; return object()
        bridge = self._bridge(ti_result=_ConnTerm(), ai_result=_ai)
        assert bridge._heartbeat_ok() is True
        assert ai_calls[0] == 0, "account_info must not be called when terminal connected"
