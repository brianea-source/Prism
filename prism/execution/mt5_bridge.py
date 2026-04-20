"""
PRISM MT5 Execution Bridge — Exness integration.
Connects to MetaTrader5 terminal via Python API.
Supports CONFIRM mode (Slack approval gated elsewhere), AUTO mode, NOTIFY mode.

Requirements:
- MetaTrader5 package: pip install MetaTrader5
- MT5 terminal running on same machine (Windows or Wine on Linux)
- Exness account credentials in environment variables
"""
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List

logger = logging.getLogger(__name__)

# ---- Constants ----
MAGIC_NUMBER = 20260420  # PRISM identifier in MT5
RISK_PCT = float(os.environ.get("PRISM_RISK_PCT", "0.01"))  # 1% per trade
MAX_CONCURRENT = int(os.environ.get("PRISM_MAX_CONCURRENT", "3"))

# Pip size per instrument
PIP_SIZE = {
    "XAUUSD": 0.01,    # Gold: 1 pip = $0.01 per oz
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
}

# APPROXIMATE retail pip values per 1.0 standard lot. These are fallbacks when
# the MT5 symbol info isn't available (offline, mock, or misconfigured). In
# production we read symbol_info.trade_tick_value / trade_tick_size from MT5.
APPROX_PIP_VALUE_PER_LOT = {
    "XAUUSD": 1.0,   # Approx; real value depends on broker contract size (usually 100 oz)
    "USDJPY": 7.0,   # Approx; varies with USD/JPY price
    "__DEFAULT__": 10.0,  # Approx for EURUSD/GBPUSD major pairs
}

# Common Exness symbol suffixes. Resolved at connect() time against
# mt5.symbols_get() so we don't hard-code broker-specific naming.
EXNESS_SUFFIX_CANDIDATES = ("", "m", "z", ".", ".m", "-m", "_m")


@dataclass
class SignalPacket:
    """Complete signal ready for MT5 execution."""
    instrument: str
    direction: str        # "LONG" or "SHORT"
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr_ratio: float
    confidence: float
    confidence_level: str
    magnitude_pips: float
    regime: str
    news_bias: str
    fvg_zone: Optional[dict]
    signal_time: str
    model_version: str = "prism_v2.0"
    # UUID stamped at construction time so every packet — whether built by the
    # SignalGenerator, a retry, a reconciliation replay, or a test fixture —
    # carries a stable audit ID without needing a post-hoc assignment.
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ExecutionResult:
    success: bool
    ticket: Optional[int]      # MT5 order ticket
    error: Optional[str]
    actual_entry: Optional[float]
    actual_sl: Optional[float]
    actual_tp: Optional[float]
    executed_at: Optional[str]
    status: str = "EXECUTED"   # EXECUTED | PENDING_APPROVAL | NOTIFY | REJECTED


class MT5Bridge:
    """
    MetaTrader5 execution bridge for PRISM.
    Handles connection, position sizing, order placement, and trade management.
    """

    def __init__(self, mode: str = "CONFIRM"):
        """
        mode: "CONFIRM" — returns PENDING_APPROVAL result; an external approval
                          surface (e.g. OpenClaw → Slack) must call submit_order()
                          to actually send the trade.
              "AUTO"    — executes immediately on signal.
              "NOTIFY"  — sends alert only, no execution.
        """
        self.mode = mode
        self._mt5 = None
        self._connected = False
        # instrument -> resolved broker symbol name (may differ from instrument)
        self._symbol_cache: dict = {}

    def connect(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
    ) -> bool:
        """Connect to MT5 terminal. Credentials from args or environment."""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5

            login = login or int(os.environ.get("MT5_LOGIN", "0"))
            password = password or os.environ.get("MT5_PASSWORD", "")
            server = server or os.environ.get("MT5_SERVER", "")
            path = path or os.environ.get("MT5_PATH", "")

            kwargs = {"login": login, "password": password, "server": server}
            if path:
                kwargs["path"] = path

            if not mt5.initialize(**kwargs):
                error = mt5.last_error()
                logger.error(f"MT5 init failed: {error}")
                return False

            info = mt5.account_info()
            if info is None:
                logger.error("MT5 account info unavailable after connect")
                return False

            logger.info(
                f"MT5 connected: account={info.login} balance={info.balance} server={info.server}"
            )
            self._connected = True
            return True

        except ImportError:
            logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
            return False
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    def disconnect(self):
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def get_account_balance(self) -> float:
        if not self._connected:
            return 0.0
        info = self._mt5.account_info()
        return info.balance if info else 0.0

    def count_open_positions(self, instrument: Optional[str] = None) -> int:
        if not self._connected:
            return 0
        symbol = self.resolve_symbol(instrument) if instrument else None
        positions = (
            self._mt5.positions_get(symbol=symbol)
            if symbol
            else self._mt5.positions_get()
        )
        return len(positions) if positions else 0

    # ------------------------------------------------------------------
    # Symbol resolution (Exness uses suffixed names like EURUSDm)
    # ------------------------------------------------------------------

    def resolve_symbol(self, instrument: str) -> str:
        """
        Resolve a PRISM instrument ID (e.g. "EURUSD") to the broker-specific
        symbol name (e.g. "EURUSDm" for Exness retail). Falls back to the
        original instrument string when not connected or no match is found.
        """
        if instrument in self._symbol_cache:
            return self._symbol_cache[instrument]
        if not self._connected or self._mt5 is None:
            return instrument
        try:
            # Fast path: exact symbol exists
            if self._mt5.symbol_info(instrument) is not None:
                self._symbol_cache[instrument] = instrument
                return instrument
            # Try known suffix candidates
            for suffix in EXNESS_SUFFIX_CANDIDATES:
                candidate = f"{instrument}{suffix}"
                if self._mt5.symbol_info(candidate) is not None:
                    self._symbol_cache[instrument] = candidate
                    logger.info(f"Resolved {instrument} → {candidate}")
                    return candidate
        except Exception as e:
            logger.warning(f"Symbol resolution failed for {instrument}: {e}")
        # No match — cache the raw name so we only log once
        self._symbol_cache[instrument] = instrument
        return instrument

    def _pick_filling_mode(self, symbol: str):
        """
        Read symbol_info.filling_mode and return a compatible order filling flag.
        filling_mode is a bitmask: bit 0 = FOK, bit 1 = IOC, bit 2 = RETURN.
        Preference order: IOC -> FOK -> RETURN. Falls back to IOC if unknown.
        """
        mt5 = self._mt5
        if mt5 is None:
            return None
        try:
            info = mt5.symbol_info(symbol)
            flags = getattr(info, "filling_mode", 0) if info is not None else 0
            # SYMBOL_FILLING_FOK = 1, SYMBOL_FILLING_IOC = 2 per MT5 docs
            if flags & 2:
                return mt5.ORDER_FILLING_IOC
            if flags & 1:
                return mt5.ORDER_FILLING_FOK
            return mt5.ORDER_FILLING_RETURN
        except Exception:
            return mt5.ORDER_FILLING_IOC

    def _pip_value_per_lot(self, symbol: str, instrument: str) -> float:
        """
        Prefer live symbol_info.trade_tick_value / trade_tick_size when connected;
        otherwise fall back to APPROX_PIP_VALUE_PER_LOT (retail approximation).
        """
        pip = PIP_SIZE.get(instrument, 0.0001)
        if self._connected and self._mt5 is not None:
            try:
                info = self._mt5.symbol_info(symbol)
                tick_value = getattr(info, "trade_tick_value", None) if info else None
                tick_size = getattr(info, "trade_tick_size", None) if info else None
                if tick_value and tick_size and tick_size > 0:
                    return float(tick_value) * (pip / float(tick_size))
            except Exception as e:
                logger.debug(f"symbol_info lookup failed for {symbol}: {e}")
        # Fallback: retail approximation
        if instrument == "XAUUSD":
            return APPROX_PIP_VALUE_PER_LOT["XAUUSD"]
        if "JPY" in instrument:
            return APPROX_PIP_VALUE_PER_LOT["USDJPY"]
        return APPROX_PIP_VALUE_PER_LOT["__DEFAULT__"]

    def calculate_lot_size(
        self,
        instrument: str,
        sl_price: float,
        entry_price: float,
        account_balance: float,
    ) -> float:
        """
        Risk-based lot sizing: risk 1% of balance on this trade.
        lot_size = (balance * risk_pct) / (sl_pips * pip_value_per_lot)

        When connected to MT5, pip_value_per_lot is derived from
        symbol_info.trade_tick_value / trade_tick_size. Offline we fall back to
        APPROX_PIP_VALUE_PER_LOT — acceptable for v0; label trades accordingly.
        """
        pip = PIP_SIZE.get(instrument, 0.0001)
        sl_pips = abs(entry_price - sl_price) / pip
        if sl_pips < 1:
            logger.warning(f"SL too tight: {sl_pips:.1f} pips — rejecting")
            return 0.0

        risk_amount = account_balance * RISK_PCT
        symbol = self.resolve_symbol(instrument)
        pip_value_per_lot = self._pip_value_per_lot(symbol, instrument)

        lot = risk_amount / (sl_pips * pip_value_per_lot)
        lot = max(0.01, min(lot, 10.0))
        lot = round(lot, 2)
        logger.info(
            f"Lot size: {lot} (risk={risk_amount:.2f}, sl={sl_pips:.1f} pips, "
            f"pip_value={pip_value_per_lot:.4f}, symbol={symbol})"
        )
        return lot

    def execute_signal(self, signal: SignalPacket) -> ExecutionResult:
        """
        Dispatch a signal according to the configured mode.

        * NOTIFY   -> never executes; returns status=NOTIFY
        * CONFIRM  -> returns status=PENDING_APPROVAL; caller must invoke
                      submit_order(signal) after human/Slack confirmation.
        * AUTO     -> calls submit_order(signal) immediately.
        """
        if self.mode == "NOTIFY":
            logger.info(
                f"NOTIFY mode: signal logged, not executed: {signal.instrument} {signal.direction}"
            )
            return ExecutionResult(
                success=False, ticket=None, error="NOTIFY mode",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="NOTIFY",
            )

        if self.mode == "CONFIRM":
            logger.info(
                f"CONFIRM mode: awaiting approval for {signal.instrument} "
                f"{signal.direction} @ {signal.entry}"
            )
            return ExecutionResult(
                success=False, ticket=None, error=None,
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="PENDING_APPROVAL",
            )

        # AUTO
        return self.submit_order(signal)

    def submit_order(self, signal: SignalPacket) -> ExecutionResult:
        """
        Actually send the order to MT5. This is the approval surface that the
        Slack/OpenClaw confirm handler calls after the user clicks "approve".
        """
        if not self._connected:
            return ExecutionResult(
                success=False, ticket=None, error="MT5 not connected",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        open_count = self.count_open_positions()
        if open_count >= MAX_CONCURRENT:
            return ExecutionResult(
                success=False, ticket=None,
                error=f"Max concurrent trades reached ({MAX_CONCURRENT})",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        if signal.confidence < 0.60:
            return ExecutionResult(
                success=False, ticket=None,
                error=f"Confidence too low: {signal.confidence:.2f}",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        mt5 = self._mt5
        balance = self.get_account_balance()
        lot = self.calculate_lot_size(signal.instrument, signal.sl, signal.entry, balance)
        if lot <= 0:
            return ExecutionResult(
                success=False, ticket=None, error="Invalid lot size",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        symbol = self.resolve_symbol(signal.instrument)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return ExecutionResult(
                success=False, ticket=None,
                error=f"No tick data for {symbol}",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        order_type = mt5.ORDER_TYPE_BUY if signal.direction == "LONG" else mt5.ORDER_TYPE_SELL
        price = tick.ask if signal.direction == "LONG" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": signal.sl,
            "tp": signal.tp2,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            # Short signal_id prefix keeps us under MT5's 31-char comment limit
            # while still being unique enough to reconcile Slack ↔ MT5 ticket.
            "comment": (
                f"PRISM_{(signal.signal_id or '')[:8]}_"
                f"{signal.direction}_{signal.confidence:.2f}"
            ),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._pick_filling_mode(symbol),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = str(result.retcode) if result else "None"
            logger.error(f"Order failed: {error}")
            return ExecutionResult(
                success=False, ticket=None, error=error,
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        logger.info(f"Order executed: ticket={result.order} price={result.price} lot={lot}")
        return ExecutionResult(
            success=True,
            ticket=result.order,
            error=None,
            actual_entry=result.price,
            actual_sl=signal.sl,
            actual_tp=signal.tp2,
            executed_at=datetime.now(timezone.utc).isoformat(),
            status="EXECUTED",
        )

    def close_position(self, ticket: int) -> bool:
        """Close a specific position by ticket number."""
        if not self._connected:
            return False
        mt5 = self._mt5
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        pos = position[0]
        tick = mt5.symbol_info_tick(pos.symbol)
        close_price = tick.bid if pos.type == 0 else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "PRISM_CLOSE",
            "type_filling": self._pick_filling_mode(pos.symbol),
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


class MockMT5Bridge(MT5Bridge):
    """
    Mock bridge for testing without a live MT5 terminal.
    Returns fake ExecutionResult — useful for backtesting signal delivery.
    """

    def connect(self, **kwargs) -> bool:
        self._connected = True
        logger.info("MockMT5Bridge: connected (no real terminal)")
        return True

    def get_account_balance(self) -> float:
        return 1000.0  # Simulated $1000 account

    def count_open_positions(self, instrument=None) -> int:
        return 0

    def resolve_symbol(self, instrument: str) -> str:
        return instrument

    def _pip_value_per_lot(self, symbol: str, instrument: str) -> float:
        # Preserve the original (approximate) retail values for deterministic tests.
        if instrument == "XAUUSD":
            return APPROX_PIP_VALUE_PER_LOT["XAUUSD"]
        if "JPY" in instrument:
            return APPROX_PIP_VALUE_PER_LOT["USDJPY"]
        return APPROX_PIP_VALUE_PER_LOT["__DEFAULT__"]

    def execute_signal(self, signal: SignalPacket) -> ExecutionResult:
        # In mock we always simulate an executed trade (regardless of mode)
        # so existing tests and backtests get deterministic outputs.
        logger.info(
            f"MockMT5Bridge: simulated execute {signal.instrument} {signal.direction} @ {signal.entry}"
        )
        return ExecutionResult(
            success=True,
            ticket=99999,
            error=None,
            actual_entry=signal.entry,
            actual_sl=signal.sl,
            actual_tp=signal.tp2,
            executed_at=datetime.now(timezone.utc).isoformat(),
            status="EXECUTED",
        )

    def submit_order(self, signal: SignalPacket) -> ExecutionResult:
        return self.execute_signal(signal)
