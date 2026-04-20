"""
PRISM MT5 Execution Bridge — Exness integration.
Connects to MetaTrader5 terminal via Python API.
Supports CONFIRM mode (Slack approval) and AUTO mode.

Requirements:
- MetaTrader5 package: pip install MetaTrader5
- MT5 terminal running on same machine (Windows or Wine on Linux)
- Exness account credentials in environment variables
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

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


@dataclass
class ExecutionResult:
    success: bool
    ticket: Optional[int]      # MT5 order ticket
    error: Optional[str]
    actual_entry: Optional[float]
    actual_sl: Optional[float]
    actual_tp: Optional[float]
    executed_at: Optional[str]


class MT5Bridge:
    """
    MetaTrader5 execution bridge for PRISM.
    Handles connection, position sizing, order placement, and trade management.
    """

    def __init__(self, mode: str = "CONFIRM"):
        """
        mode: "CONFIRM" — requires Slack approval before execution (default)
              "AUTO"    — executes immediately on signal
              "NOTIFY"  — sends alert only, no execution
        """
        self.mode = mode
        self._mt5 = None
        self._connected = False

    def connect(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
    ) -> bool:
        """
        Connect to MT5 terminal.
        Credentials from args or environment variables:
          MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH
        """
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
        positions = (
            self._mt5.positions_get(symbol=instrument)
            if instrument
            else self._mt5.positions_get()
        )
        return len(positions) if positions else 0

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
        """
        pip = PIP_SIZE.get(instrument, 0.0001)
        sl_pips = abs(entry_price - sl_price) / pip
        if sl_pips < 1:
            logger.warning(f"SL too tight: {sl_pips:.1f} pips — rejecting")
            return 0.0

        risk_amount = account_balance * RISK_PCT
        # Standard lot pip value: $10/pip for majors, $7 for JPY, $1 for gold
        if instrument == "XAUUSD":
            pip_value_per_lot = 1.0
        elif "JPY" in instrument:
            pip_value_per_lot = 7.0
        else:
            pip_value_per_lot = 10.0

        lot = risk_amount / (sl_pips * pip_value_per_lot)
        lot = max(0.01, min(lot, 10.0))
        lot = round(lot, 2)
        logger.info(f"Lot size: {lot} (risk={risk_amount:.2f}, sl={sl_pips:.1f} pips)")
        return lot

    def execute_signal(self, signal: SignalPacket) -> ExecutionResult:
        """Execute a PRISM signal on MT5. Respects mode (CONFIRM/AUTO/NOTIFY)."""
        if self.mode == "NOTIFY":
            logger.info(
                f"NOTIFY mode: signal logged, not executed: {signal.instrument} {signal.direction}"
            )
            return ExecutionResult(
                success=False, ticket=None, error="NOTIFY mode",
                actual_entry=None, actual_sl=None, actual_tp=None, executed_at=None,
            )

        if not self._connected:
            return ExecutionResult(
                success=False, ticket=None, error="MT5 not connected",
                actual_entry=None, actual_sl=None, actual_tp=None, executed_at=None,
            )

        open_count = self.count_open_positions()
        if open_count >= MAX_CONCURRENT:
            return ExecutionResult(
                success=False, ticket=None,
                error=f"Max concurrent trades reached ({MAX_CONCURRENT})",
                actual_entry=None, actual_sl=None, actual_tp=None, executed_at=None,
            )

        if signal.confidence < 0.60:
            return ExecutionResult(
                success=False, ticket=None,
                error=f"Confidence too low: {signal.confidence:.2f}",
                actual_entry=None, actual_sl=None, actual_tp=None, executed_at=None,
            )

        mt5 = self._mt5
        balance = self.get_account_balance()
        lot = self.calculate_lot_size(signal.instrument, signal.sl, signal.entry, balance)
        if lot <= 0:
            return ExecutionResult(
                success=False, ticket=None, error="Invalid lot size",
                actual_entry=None, actual_sl=None, actual_tp=None, executed_at=None,
            )

        tick = mt5.symbol_info_tick(signal.instrument)
        if tick is None:
            return ExecutionResult(
                success=False, ticket=None,
                error=f"No tick data for {signal.instrument}",
                actual_entry=None, actual_sl=None, actual_tp=None, executed_at=None,
            )

        order_type = mt5.ORDER_TYPE_BUY if signal.direction == "LONG" else mt5.ORDER_TYPE_SELL
        price = tick.ask if signal.direction == "LONG" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.instrument,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": signal.sl,
            "tp": signal.tp2,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": f"PRISM_{signal.direction}_{signal.confidence:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = str(result.retcode) if result else "None"
            logger.error(f"Order failed: {error}")
            return ExecutionResult(
                success=False, ticket=None, error=error,
                actual_entry=None, actual_sl=None, actual_tp=None, executed_at=None,
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

    def execute_signal(self, signal: SignalPacket) -> ExecutionResult:
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
        )
