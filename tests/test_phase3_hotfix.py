"""
PRISM Phase 3 hotfix regression tests.

Covers the three blockers that were hiding behind MockMT5Bridge:

1. runner._scan_instrument CONFIRM flow must dispatch via
   bridge.submit_order() after user approval -- bridge.execute_signal()
   in CONFIRM mode deliberately returns PENDING_APPROVAL and would
   flag every confirmed signal as FAILED on a real MT5 terminal.
2. runner._resolve_cache_paths must glob on INSTRUMENT_MAP[symbol], not
   the symbol string itself. Otherwise XAUUSD (which caches as
   tiingo_GLD_*.parquet) silently never produces a signal.
3. slack-sdk must be declared in requirements.txt; slack_notifier
   imports it at module top.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest import mock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared lightweight signal stand-in (SignalPacket has many required fields)
# ---------------------------------------------------------------------------
@dataclass
class StubSignal:
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


@dataclass
class StubExecResult:
    success: bool
    ticket: Optional[int] = 12345
    error: Optional[str] = None
    actual_entry: Optional[float] = None
    actual_sl: Optional[float] = None
    actual_tp: Optional[float] = None
    executed_at: Optional[str] = None
    status: str = "EXECUTED"


class StubBridge:
    """Bridge spy that records which dispatch method runner chose."""

    def __init__(
        self,
        mode: str = "CONFIRM",
        submit_result: Optional[StubExecResult] = None,
        execute_result: Optional[StubExecResult] = None,
    ):
        self.mode = mode
        self.submit_calls: list = []
        self.execute_calls: list = []
        self._submit_result = submit_result or StubExecResult(success=True)
        # Mirror real MT5Bridge: execute_signal in CONFIRM must not place an order.
        self._execute_result = execute_result or (
            StubExecResult(success=False, ticket=None, status="PENDING_APPROVAL",
                           error=None)
            if mode == "CONFIRM"
            else StubExecResult(success=True)
        )

    def submit_order(self, signal):
        self.submit_calls.append(signal)
        return self._submit_result

    def execute_signal(self, signal):
        self.execute_calls.append(signal)
        return self._execute_result


class StubNotifier:
    def __init__(self):
        self.channel = "#test-signals"
        self.sent: list = []
        self.statuses: list = []

    def send_signal(self, signal, mode="CONFIRM", use_buttons=False):
        self.sent.append((signal, mode))
        return "1234567890.123456"

    def update_signal_status(self, ts, status, signal):
        self.statuses.append((ts, status))


# ---------------------------------------------------------------------------
# Blocker #2: _resolve_cache_paths must honour INSTRUMENT_MAP
# ---------------------------------------------------------------------------
def _resolved_names(paths) -> set:
    """Compare by filename to avoid absolute-vs-relative path coupling."""
    return {Path(p).name for p in paths}


class TestCachePathResolution:
    def test_xauusd_resolves_to_gld_cache_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True)
        expected_name = "tiingo_GLD_4hour_2024-01-01_2024-01-05.parquet"
        (tmp_path / "data" / "raw" / expected_name).touch()

        from prism.delivery.runner import _resolve_cache_paths
        paths = _resolve_cache_paths("XAUUSD", "4hour")
        assert expected_name in _resolved_names(paths), (
            "XAUUSD must resolve via INSTRUMENT_MAP -> 'GLD'; globbing on the "
            "symbol name itself would never match the real cache file."
        )

    def test_eurusd_uppercase_ticker_cache_matches(self, tmp_path, monkeypatch):
        """
        Current main has INSTRUMENT_MAP['EURUSD'] = 'EURUSD' (uppercase), so
        cache files are named tiingo_EURUSD_4hour_*.parquet. On Linux this is
        case-sensitive; we must match both cases.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True)
        expected_name = "tiingo_EURUSD_4hour_2024-01-01_2024-01-05.parquet"
        (tmp_path / "data" / "raw" / expected_name).touch()

        from prism.delivery.runner import _resolve_cache_paths
        assert expected_name in _resolved_names(_resolve_cache_paths("EURUSD", "4hour"))

    def test_eurusd_lowercase_ticker_cache_also_matches(self, tmp_path, monkeypatch):
        """
        PR #4 flips INSTRUMENT_MAP['EURUSD'] to 'eurusd' (lowercase, per Tiingo
        FX docs). Resolver must still find the cache after that merge.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True)
        expected_name = "tiingo_eurusd_4hour_2024-01-01_2024-01-05.parquet"
        (tmp_path / "data" / "raw" / expected_name).touch()

        from prism.delivery.runner import _resolve_cache_paths
        assert expected_name in _resolved_names(_resolve_cache_paths("EURUSD", "4hour"))

    def test_returns_empty_when_no_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True)

        from prism.delivery.runner import _resolve_cache_paths
        assert _resolve_cache_paths("XAUUSD", "4hour") == []

    def test_unknown_instrument_falls_back_to_symbol_name(self, tmp_path, monkeypatch):
        """Unknown instruments glob on the raw symbol; they won't be in
        INSTRUMENT_MAP but callers shouldn't crash."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True)
        expected_name = "tiingo_BTCUSD_4hour_2024-01-01_2024-01-05.parquet"
        (tmp_path / "data" / "raw" / expected_name).touch()

        from prism.delivery.runner import _resolve_cache_paths
        assert expected_name in _resolved_names(_resolve_cache_paths("BTCUSD", "4hour"))


# ---------------------------------------------------------------------------
# Blocker #1: CONFIRM flow dispatches via submit_order
# ---------------------------------------------------------------------------
def _write_fake_parquet(tmp_path: Path, ticker: str = "GLD", timeframe: str = "4hour") -> None:
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    # Tiny OHLCV frame; SignalGenerator is patched out, but parquet must be
    # readable by pandas to exercise the real code path up to that point.
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC"),
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.5] * 5,
        "volume": [1000.0] * 5,
    })
    df.to_parquet(tmp_path / "data" / "raw" / f"tiingo_{ticker}_{timeframe}_2024-01-01_2024-01-05.parquet")


class TestConfirmDispatch:
    def test_confirm_confirmed_calls_submit_order_not_execute_signal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_fake_parquet(tmp_path, ticker="GLD")

        bridge = StubBridge(mode="CONFIRM")
        notifier = StubNotifier()

        with mock.patch("prism.signal.generator.SignalGenerator") as gen_cls, \
             mock.patch("prism.delivery.confirm_handler.PollConfirmHandler") as handler_cls, \
             mock.patch("slack_sdk.WebClient"):

            gen_cls.return_value.generate.return_value = StubSignal()
            handler_cls.return_value.wait.return_value = "CONFIRMED"

            from prism.delivery.runner import _scan_instrument
            _scan_instrument("XAUUSD", notifier, bridge, datetime.now(timezone.utc))

        assert len(bridge.submit_calls) == 1, (
            "CONFIRM+CONFIRMED must dispatch via submit_order; execute_signal "
            "in CONFIRM mode returns PENDING_APPROVAL and would mark the "
            "trade FAILED."
        )
        assert bridge.execute_calls == []
        assert ("1234567890.123456", "EXECUTED") in notifier.statuses

    def test_confirm_skipped_does_not_execute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_fake_parquet(tmp_path, ticker="GLD")

        bridge = StubBridge(mode="CONFIRM")
        notifier = StubNotifier()

        with mock.patch("prism.signal.generator.SignalGenerator") as gen_cls, \
             mock.patch("prism.delivery.confirm_handler.PollConfirmHandler") as handler_cls, \
             mock.patch("slack_sdk.WebClient"):

            gen_cls.return_value.generate.return_value = StubSignal()
            handler_cls.return_value.wait.return_value = "SKIPPED"

            from prism.delivery.runner import _scan_instrument
            _scan_instrument("XAUUSD", notifier, bridge, datetime.now(timezone.utc))

        assert bridge.submit_calls == []
        assert bridge.execute_calls == []
        assert ("1234567890.123456", "SKIPPED") in notifier.statuses

    def test_confirm_expired_does_not_execute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_fake_parquet(tmp_path, ticker="GLD")

        bridge = StubBridge(mode="CONFIRM")
        notifier = StubNotifier()

        with mock.patch("prism.signal.generator.SignalGenerator") as gen_cls, \
             mock.patch("prism.delivery.confirm_handler.PollConfirmHandler") as handler_cls, \
             mock.patch("slack_sdk.WebClient"):

            gen_cls.return_value.generate.return_value = StubSignal()
            handler_cls.return_value.wait.return_value = "EXPIRED"

            from prism.delivery.runner import _scan_instrument
            _scan_instrument("XAUUSD", notifier, bridge, datetime.now(timezone.utc))

        assert bridge.submit_calls == []
        assert bridge.execute_calls == []
        assert ("1234567890.123456", "EXPIRED") in notifier.statuses

    def test_auto_mode_dispatches_via_execute_signal(self, tmp_path, monkeypatch):
        """AUTO mode should go through execute_signal, which the bridge
        internally routes to submit_order. Runner should NOT double-send."""
        monkeypatch.chdir(tmp_path)
        _write_fake_parquet(tmp_path, ticker="GLD")

        bridge = StubBridge(mode="AUTO",
                            execute_result=StubExecResult(success=True, status="EXECUTED"))
        notifier = StubNotifier()

        with mock.patch("prism.signal.generator.SignalGenerator") as gen_cls, \
             mock.patch("slack_sdk.WebClient"):

            gen_cls.return_value.generate.return_value = StubSignal()

            from prism.delivery.runner import _scan_instrument
            _scan_instrument("XAUUSD", notifier, bridge, datetime.now(timezone.utc))

        assert len(bridge.execute_calls) == 1
        assert bridge.submit_calls == []

    def test_no_signal_generated_skips_notifier(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_fake_parquet(tmp_path, ticker="GLD")

        bridge = StubBridge(mode="CONFIRM")
        notifier = StubNotifier()

        with mock.patch("prism.signal.generator.SignalGenerator") as gen_cls:
            gen_cls.return_value.generate.return_value = None

            from prism.delivery.runner import _scan_instrument
            _scan_instrument("XAUUSD", notifier, bridge, datetime.now(timezone.utc))

        assert notifier.sent == []
        assert bridge.submit_calls == []
        assert bridge.execute_calls == []

    def test_missing_cache_returns_without_generator(self, tmp_path, monkeypatch):
        """No cache files for instrument -> no SignalGenerator call, no Slack, no bridge."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True)  # empty

        bridge = StubBridge(mode="CONFIRM")
        notifier = StubNotifier()

        with mock.patch("prism.signal.generator.SignalGenerator") as gen_cls:
            from prism.delivery.runner import _scan_instrument
            _scan_instrument("XAUUSD", notifier, bridge, datetime.now(timezone.utc))

            gen_cls.assert_not_called()

        assert notifier.sent == []
        assert bridge.submit_calls == []
        assert bridge.execute_calls == []


# ---------------------------------------------------------------------------
# Blocker #3: slack-sdk pinned in requirements.txt
# ---------------------------------------------------------------------------
class TestRequirements:
    def test_requirements_declares_slack_sdk(self):
        root = Path(__file__).resolve().parent.parent
        req = (root / "requirements.txt").read_text()
        assert "slack-sdk" in req or "slack_sdk" in req, (
            "slack_notifier imports slack_sdk at module top; without a pinned "
            "dependency, a fresh `pip install -r requirements.txt` followed by "
            "`python prism/delivery/runner.py` raises ModuleNotFoundError."
        )
