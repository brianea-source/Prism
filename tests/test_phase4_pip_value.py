"""
PRISM Phase 4 — real pip value from MT5 symbol_info.

The retail approximation (`$10/EURUSD, $7/JPY, $1/XAUUSD per lot per pip`)
is up to ~20% off on XAUUSD and JPY-quoted pairs depending on the broker's
contract size and the account currency. On a live account this means the
advertised "1% risk" silently becomes 0.8-1.2% risk. This test file locks
in three properties:

1. When MT5 is connected, pip value comes from ``symbol_info`` (live path).
2. Field priority: ``trade_tick_value_loss`` > ``trade_tick_value`` >
   ``trade_tick_value_profit``. Loss-side is the correct risk basis.
3. When ``symbol_info`` returns nothing on a live bridge, the trade is
   REJECTED (lot=0) rather than silently sized from the retail approximation.
   Opt-in via ``PRISM_ALLOW_APPROX_PIP_VALUE=1`` for demo/dev only.
"""
from __future__ import annotations

import os

import pytest

from prism.execution.mt5_bridge import (
    APPROX_PIP_VALUE_PER_LOT,
    MockMT5Bridge,
    MT5Bridge,
    PIP_SIZE,
    RISK_PCT,
)


# ---------------------------------------------------------------------------
# Fake symbol_info object — only exposes the tick-value / tick-size fields
# the bridge reads. Attributes default to None; tests set what they need.
# ---------------------------------------------------------------------------

class _FakeSymbolInfo:
    def __init__(self, **kw):
        self.trade_tick_size = kw.get("trade_tick_size")
        self.trade_tick_value = kw.get("trade_tick_value")
        self.trade_tick_value_loss = kw.get("trade_tick_value_loss")
        self.trade_tick_value_profit = kw.get("trade_tick_value_profit")


class _FakeMt5:
    def __init__(self, info: _FakeSymbolInfo):
        self._info = info

    def symbol_info(self, _symbol):
        return self._info


def _connected_bridge(info: _FakeSymbolInfo) -> MT5Bridge:
    b = MT5Bridge(mode="CONFIRM")
    b._connected = True
    b._mt5 = _FakeMt5(info)
    return b


# ===========================================================================
# Field priority
# ===========================================================================

class TestFieldPriority:
    def test_prefers_trade_tick_value_loss(self):
        """
        When all three fields are present, *_loss wins. Loss-side is what
        we want for risk dimensioning — the position is sized against the
        actual loss at SL, not the slightly-different swap-inclusive profit.
        """
        info = _FakeSymbolInfo(
            trade_tick_size=0.0001,
            trade_tick_value=11.0,          # would pick this if loss absent
            trade_tick_value_loss=10.0,     # winner
            trade_tick_value_profit=12.0,   # last resort
        )
        bridge = _connected_bridge(info)
        val, source = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert source == "symbol_info"
        # pip=0.0001, tick_size=0.0001 → ratio=1 → value == tick_value_loss
        assert val == pytest.approx(10.0)

    def test_falls_back_to_trade_tick_value_when_loss_missing(self):
        info = _FakeSymbolInfo(
            trade_tick_size=0.0001,
            trade_tick_value=11.0,
            trade_tick_value_profit=12.0,
            # trade_tick_value_loss not set
        )
        bridge = _connected_bridge(info)
        val, source = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert source == "symbol_info"
        assert val == pytest.approx(11.0)

    def test_falls_back_to_profit_when_loss_and_value_missing(self):
        info = _FakeSymbolInfo(
            trade_tick_size=0.0001,
            trade_tick_value_profit=12.0,
        )
        bridge = _connected_bridge(info)
        val, source = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert source == "symbol_info"
        assert val == pytest.approx(12.0)

    def test_zero_loss_is_ignored(self):
        """
        Some brokers populate *_loss with 0 by accident. Treat zero as 'not
        populated' and fall through to the next field rather than divide
        the lot size by zero downstream.
        """
        info = _FakeSymbolInfo(
            trade_tick_size=0.0001,
            trade_tick_value=11.0,
            trade_tick_value_loss=0.0,
        )
        bridge = _connected_bridge(info)
        val, source = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert source == "symbol_info"
        assert val == pytest.approx(11.0)


# ===========================================================================
# Pip-value math — XAUUSD vs EURUSD vs USDJPY
# ===========================================================================

class TestPipValueMath:
    def test_eurusd_unit_ratio(self):
        """EURUSD: pip=0.0001, tick_size=0.0001 → pip_value == tick_value."""
        info = _FakeSymbolInfo(
            trade_tick_size=0.0001,
            trade_tick_value_loss=10.05,  # real Exness-ish value
        )
        bridge = _connected_bridge(info)
        val, _ = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert val == pytest.approx(10.05)

    def test_xauusd_scales_correctly(self):
        """
        XAUUSD pip_size is 0.01. If broker reports tick_size=0.01 and
        tick_value_loss=1.00 per 0.01 tick, pip_value = 1.00. If broker
        uses tick_size=0.10 (some retail brokers do), pip_value should
        scale to 0.10 — that's the whole point of the ratio.
        """
        # Common Exness-style XAUUSD: tick_size=0.01, tick_value=1.0 USD per
        # 0.01 move per 1 lot (100 oz contract).
        info = _FakeSymbolInfo(
            trade_tick_size=0.01,
            trade_tick_value_loss=1.0,
        )
        bridge = _connected_bridge(info)
        val, _ = bridge._pip_value_per_lot("XAUUSDm", "XAUUSD")
        assert val == pytest.approx(1.0)

        # Alternative scale — tick_size=0.10 means each reported tick is 10
        # pips, so per-pip value is 1/10 of reported tick value.
        info2 = _FakeSymbolInfo(
            trade_tick_size=0.10,
            trade_tick_value_loss=10.0,
        )
        bridge2 = _connected_bridge(info2)
        val2, _ = bridge2._pip_value_per_lot("XAUUSDm", "XAUUSD")
        assert val2 == pytest.approx(1.0)

    def test_usdjpy_scales_correctly(self):
        """
        USDJPY pip_size is 0.01 (yen pairs). tick_size typically 0.001 so
        the ratio is 10 — a broker reporting $0.73 per tick means $7.30 per pip.
        """
        info = _FakeSymbolInfo(
            trade_tick_size=0.001,
            trade_tick_value_loss=0.73,
        )
        bridge = _connected_bridge(info)
        val, _ = bridge._pip_value_per_lot("USDJPYm", "USDJPY")
        assert val == pytest.approx(7.3)


# ===========================================================================
# Source contract + guard against missing symbol_info
# ===========================================================================

class TestSourceContract:
    def test_symbol_info_source_label(self):
        info = _FakeSymbolInfo(trade_tick_size=0.0001, trade_tick_value=10.0)
        bridge = _connected_bridge(info)
        _, source = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert source == "symbol_info"

    def test_unavailable_when_connected_but_no_info(self):
        """
        Broker returned None for symbol_info (bad symbol map, network blip).
        On a connected bridge we must report "unavailable" — not silently
        hand back the retail approximation for a live trade.
        """
        class _NullInfoMt5:
            def symbol_info(self, _s): return None
        bridge = MT5Bridge(mode="CONFIRM")
        bridge._connected = True
        bridge._mt5 = _NullInfoMt5()
        val, source = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert source == "unavailable"
        assert val == 0.0

    def test_unavailable_when_tick_size_zero(self):
        info = _FakeSymbolInfo(trade_tick_size=0.0, trade_tick_value=10.0)
        bridge = _connected_bridge(info)
        val, source = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert source == "unavailable"
        assert val == 0.0

    def test_unavailable_when_all_tick_values_zero(self):
        info = _FakeSymbolInfo(
            trade_tick_size=0.0001,
            trade_tick_value=0.0,
            trade_tick_value_loss=0.0,
            trade_tick_value_profit=0.0,
        )
        bridge = _connected_bridge(info)
        val, source = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert source == "unavailable"
        assert val == 0.0

    def test_disconnected_falls_back_to_approximation(self):
        """Offline bridge = unit-test / dev mode; approximation is fine here."""
        bridge = MT5Bridge(mode="CONFIRM")
        val, source = bridge._pip_value_per_lot("EURUSD", "EURUSD")
        assert source == "approximation"
        assert val == pytest.approx(APPROX_PIP_VALUE_PER_LOT["__DEFAULT__"])

    def test_symbol_info_raises_falls_through_to_approx_or_unavailable(self):
        class _RaisingMt5:
            def symbol_info(self, _s):
                raise RuntimeError("mt5 crashed")

        # Connected: exception → unavailable (the safe answer)
        bridge = MT5Bridge(mode="CONFIRM")
        bridge._connected = True
        bridge._mt5 = _RaisingMt5()
        val, source = bridge._pip_value_per_lot("EURUSDm", "EURUSD")
        assert source == "unavailable"
        assert val == 0.0


# ===========================================================================
# calculate_lot_size end-to-end — the production safety behaviour
# ===========================================================================

class TestCalculateLotSize:
    def test_live_symbol_info_sizes_at_1pct_risk(self):
        """
        Real-numbers end-to-end. $10,000 account, 1% risk = $100.
        EURUSD SL 50 pips. Broker reports $10/pip → lot = 100/(50*10) = 0.20.
        """
        info = _FakeSymbolInfo(
            trade_tick_size=0.0001,
            trade_tick_value_loss=10.0,
        )
        bridge = _connected_bridge(info)
        lot = bridge.calculate_lot_size(
            instrument="EURUSD",
            entry_price=1.1000,
            sl_price=1.0950,  # 50 pips
            account_balance=10_000.0,
        )
        expected = round((10_000.0 * RISK_PCT) / (50.0 * 10.0), 2)
        assert lot == pytest.approx(expected)

    def test_rejects_when_symbol_info_unavailable_on_live_bridge(self, monkeypatch):
        """
        The headline behaviour: no symbol_info + live bridge + no opt-in =
        lot 0 = trade rejected. Better to skip a signal than risk 20% off on
        a real account.
        """
        monkeypatch.delenv("PRISM_ALLOW_APPROX_PIP_VALUE", raising=False)
        class _NullInfo:
            def symbol_info(self, _s): return None
        bridge = MT5Bridge(mode="CONFIRM")
        bridge._connected = True
        bridge._mt5 = _NullInfo()
        lot = bridge.calculate_lot_size(
            instrument="EURUSD",
            entry_price=1.10, sl_price=1.095,
            account_balance=10_000.0,
        )
        assert lot == 0.0

    def test_opt_in_allows_approximation_on_live_bridge(self, monkeypatch):
        """
        PRISM_ALLOW_APPROX_PIP_VALUE=1 lets dev/demo runs keep sizing even
        when symbol_info is missing. Explicit, opt-in, and logged as
        source="approximation-forced" for audit.
        """
        monkeypatch.setenv("PRISM_ALLOW_APPROX_PIP_VALUE", "1")
        class _NullInfo:
            def symbol_info(self, _s): return None
        bridge = MT5Bridge(mode="CONFIRM")
        bridge._connected = True
        bridge._mt5 = _NullInfo()
        lot = bridge.calculate_lot_size(
            instrument="EURUSD",
            entry_price=1.10, sl_price=1.095,
            account_balance=10_000.0,
        )
        # Approx: $10/pip, 50 pips, $100 risk → 0.20
        assert lot == pytest.approx(0.20)

    def test_xauusd_correctness_regression(self):
        """
        XAUUSD was specifically called out for drift. With $10,000 account,
        1% risk = $100, SL 500 pips (=$5 move on gold), broker reports
        $1/pip → lot = 100/(500*1) = 0.20.

        Under the retail approximation this comes out the same only because
        XAUUSD hits the $1 fallback. The live path confirms we're computing
        from tick fields, not from the hard-coded default.
        """
        info = _FakeSymbolInfo(
            trade_tick_size=0.01,
            trade_tick_value_loss=1.0,
        )
        bridge = _connected_bridge(info)
        lot = bridge.calculate_lot_size(
            instrument="XAUUSD",
            entry_price=2000.00, sl_price=1995.00,   # 500 pips on gold
            account_balance=10_000.0,
        )
        assert lot == pytest.approx(0.20)

    def test_tight_sl_still_rejected(self):
        """Existing safety: SL < 1 pip still rejects, independent of source."""
        info = _FakeSymbolInfo(trade_tick_size=0.0001, trade_tick_value_loss=10.0)
        bridge = _connected_bridge(info)
        lot = bridge.calculate_lot_size(
            instrument="EURUSD",
            entry_price=1.10000, sl_price=1.09995,  # 0.5 pips
            account_balance=10_000.0,
        )
        assert lot == 0.0

    def test_mock_bridge_still_sizes_with_approximation(self):
        """
        MockMT5Bridge is the demo-mode path. It must keep sizing without
        symbol_info so Brian can run locally without a live terminal.
        """
        mock = MockMT5Bridge(mode="CONFIRM")
        mock.connect()
        lot = mock.calculate_lot_size(
            instrument="EURUSD",
            entry_price=1.10, sl_price=1.095,  # 50 pips
            account_balance=1_000.0,
        )
        # $10 risk / (50 pips * $10/pip) = 0.02, clamped to min 0.02
        assert lot == pytest.approx(0.02)


# ===========================================================================
# Audit logging — source label must appear in the log line
# ===========================================================================

class TestAuditLogging:
    def test_log_includes_source_symbol_info(self, caplog):
        import logging
        caplog.set_level(logging.INFO, logger="prism.execution.mt5_bridge")
        info = _FakeSymbolInfo(trade_tick_size=0.0001, trade_tick_value_loss=10.0)
        bridge = _connected_bridge(info)
        bridge.calculate_lot_size("EURUSD", 1.095, 1.10, 10_000.0)
        assert any("source=symbol_info" in r.message for r in caplog.records), \
            "Lot size log line must declare source=symbol_info for audit"

    def test_log_includes_source_approximation(self, caplog):
        import logging
        caplog.set_level(logging.INFO, logger="prism.execution.mt5_bridge")
        MockMT5Bridge(mode="CONFIRM").calculate_lot_size(
            "EURUSD", 1.095, 1.10, 1_000.0,
        )
        assert any("source=approximation" in r.message for r in caplog.records), \
            "Mock bridge must declare source=approximation for audit"


# ===========================================================================
# Fix 5 — GBPUSD/EURUSD approx fallback contract (explicit coverage)
# ===========================================================================

class TestApproxFallback:
    """Lock in the APPROX_PIP_VALUE_PER_LOT dispatch contract for MockMT5Bridge.

    GBPUSD has no explicit key in APPROX_PIP_VALUE_PER_LOT — it falls through
    to ``__DEFAULT__ = 10.0``. This test makes that contract explicit so any
    future change to the dispatch logic is immediately visible.
    """

    def test_eurusd_uses_default(self):
        b = MockMT5Bridge()
        b.connect()
        val, source = b._pip_value_per_lot("EURUSD", "EURUSD")
        assert source == "approximation"
        assert val == APPROX_PIP_VALUE_PER_LOT["__DEFAULT__"]

    def test_gbpusd_uses_default(self):
        b = MockMT5Bridge()
        b.connect()
        val, source = b._pip_value_per_lot("GBPUSD", "GBPUSD")
        assert source == "approximation"
        assert val == APPROX_PIP_VALUE_PER_LOT["__DEFAULT__"]

    def test_xauusd_uses_gold_approx(self):
        b = MockMT5Bridge()
        b.connect()
        val, source = b._pip_value_per_lot("XAUUSD", "XAUUSD")
        assert source == "approximation"
        assert val == APPROX_PIP_VALUE_PER_LOT["XAUUSD"]
