"""
prism/model/evaluate.py
PRISM Backtester — vectorized simulation of SL/TP hits on signal output.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SL_PIPS = 20
DEFAULT_TP_MULTIPLIER = 2.0   # TP = 2 × SL (default RR)


def backtest_signals(
    df: pd.DataFrame,
    signals: list[dict[str, Any]],
    instrument: str,
    initial_balance: float = 10_000.0,
    risk_pct: float = 0.01,
) -> dict[str, Any]:
    """
    Vectorized backtest that simulates SL/TP hits on future OHLC bars.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC price data with columns: open, high, low, close.
        Index must be aligned to signals (same bar ordering).
    signals : list[dict]
        One dict per bar containing at minimum:
            direction      : int   — -1/0/1
            confidence     : float — 0.0 to 1.0 (used to scale position size)
            magnitude_pips : float — expected move (used as TP distance)
    instrument : str
        Instrument name (used to determine pip size).
    initial_balance : float
        Starting account equity in USD.
    risk_pct : float
        Fraction of balance risked per trade (default 1%).

    Returns
    -------
    dict with keys:
        total_trades, win_rate, profit_factor, avg_rr,
        max_drawdown, sharpe, final_balance, total_return_pct
    """
    pip_size = 0.01 if any(x in instrument for x in ["XAU", "JPY"]) else 0.0001

    assert len(df) >= len(signals), (
        f"df has {len(df)} rows but {len(signals)} signals provided"
    )

    balance = initial_balance
    equity_curve: list[float] = [balance]
    trade_pnls: list[float] = []
    winners = 0

    for i, sig in enumerate(signals):
        direction = sig.get("direction", 0)
        if direction == 0:
            equity_curve.append(balance)
            continue

        confidence = float(sig.get("confidence", 0.5))
        magnitude = float(sig.get("magnitude_pips", DEFAULT_SL_PIPS))

        sl_pips = max(DEFAULT_SL_PIPS, magnitude * 0.5)
        tp_pips = sl_pips * DEFAULT_TP_MULTIPLIER

        sl_dist = sl_pips * pip_size
        tp_dist = tp_pips * pip_size

        entry_price = df["close"].iloc[i]
        if direction == 1:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # Risk-adjusted position sizing (confidence scales risk)
        risk_amount = balance * risk_pct * confidence
        lot_pnl_per_pip = risk_amount / sl_pips

        # Simulate: walk next N bars to find SL/TP hit
        hit = None
        rr = 0.0
        lookforward = min(20, len(df) - i - 1)
        for j in range(1, lookforward + 1):
            bar = df.iloc[i + j]
            if direction == 1:
                if bar["low"] <= sl_price:
                    hit = "sl"
                    break
                if bar["high"] >= tp_price:
                    hit = "tp"
                    break
            else:  # direction == -1
                if bar["high"] >= sl_price:
                    hit = "sl"
                    break
                if bar["low"] <= tp_price:
                    hit = "tp"
                    break

        if hit == "tp":
            pnl = lot_pnl_per_pip * tp_pips
            winners += 1
            rr = tp_pips / sl_pips
        elif hit == "sl":
            pnl = -lot_pnl_per_pip * sl_pips
            rr = -(sl_pips / sl_pips)
        else:
            # Time-expired: treat as scratch (close at current price if available)
            if i + lookforward < len(df):
                exit_price = df["close"].iloc[i + lookforward]
                if direction == 1:
                    move_pips = (exit_price - entry_price) / pip_size
                else:
                    move_pips = (entry_price - exit_price) / pip_size
                pnl = lot_pnl_per_pip * move_pips
                rr = move_pips / sl_pips
                if pnl > 0:
                    winners += 1
            else:
                pnl = 0.0
                rr = 0.0

        balance += pnl
        trade_pnls.append(pnl)
        equity_curve.append(balance)

    # --- Metrics ---
    total_trades = len(trade_pnls)
    win_rate = winners / total_trades if total_trades > 0 else 0.0

    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_rr = float(np.mean([p / (initial_balance * risk_pct) for p in trade_pnls])) if trade_pnls else 0.0

    # Drawdown
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    # Sharpe (annualised, assuming ~252 trading days, rough)
    if len(trade_pnls) > 1:
        ret_arr = np.array(trade_pnls) / initial_balance
        sharpe = float(np.mean(ret_arr) / (np.std(ret_arr) + 1e-9) * np.sqrt(252))
    else:
        sharpe = 0.0

    final_balance = balance
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    metrics = {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_rr": round(avg_rr, 4),
        "max_drawdown": round(max_drawdown, 4),
        "sharpe": round(sharpe, 4),
        "final_balance": round(final_balance, 2),
        "total_return_pct": round(total_return_pct, 4),
    }

    logger.info(
        f"[Backtest:{instrument}] trades={total_trades} win={win_rate:.1%} "
        f"PF={profit_factor:.2f} dd={max_drawdown:.1%} sharpe={sharpe:.2f} "
        f"ret={total_return_pct:.2f}%"
    )

    return metrics
