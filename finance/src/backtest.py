"""
backtest.py — Event-driven backtest engine

Key fixes vs the original:
  • Equity curve now has one value per bar (not just per trade), so the
    time-series is properly aligned for drawdown and Sharpe calculations.
  • Stop-loss is now ATR-based (dynamic) rather than a fixed percentage.
  • Position sizing respects the risk module (ATR fractional or fixed).
  • Daily loss circuit breaker prevents compounding intraday blowups.
"""

import numpy as np
from src.risk import (
    atr_position_size,
    compute_atr_stop,
    compute_trailing_stop,
    check_daily_loss_limit,
)


def backtest(
    df,
    capital=100000,
    brokerage=0.0005,       # 0.05% per leg (realistic for NSE intraday)
    slippage=0.0005,        # 0.05% per leg
    risk_per_trade=0.01,    # risk 1% of equity per trade
    atr_stop_mult=2.0,      # hard stop = 2× ATR below entry
    trailing_atr_mult=1.5,  # trailing stop = 1.5× ATR below highest
    max_daily_loss_pct=0.02, # daily loss limit = 2% of equity
):
    """
    Simulates the strategy on out-of-sample signal data.

    Returns
    -------
    trades      : list of dicts with entry/exit details
    equity_curve: list of equity values, one per bar (length == len(df))
    drawdowns   : list of drawdown values, one per bar
    """
    equity = capital
    position = 0
    entry_price = 0.0
    shares = 0
    highest_price = 0.0
    hard_stop = 0.0

    equity_curve = []
    drawdowns = []
    trades = []
    peak = capital

    daily_pnl = 0.0
    current_day = None

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']
        signal = row['signal']
        atr_val = row.get('atr', price * 0.005)   # fallback: 0.5% of price

        # ── Daily reset ────────────────────────────────────────────────────
        bar_day = row['timestamp'].date() if hasattr(row['timestamp'], 'date') else None
        if bar_day != current_day:
            daily_pnl = 0.0
            current_day = bar_day

        # ── Circuit breaker ────────────────────────────────────────────────
        # If we've lost too much today, flatten any open position and stop
        trading_halted = check_daily_loss_limit(daily_pnl, equity, max_daily_loss_pct)

        # ── EXIT LOGIC ─────────────────────────────────────────────────────
        if position == 1:
            if price > highest_price:
                highest_price = price

            trailing_stop = compute_trailing_stop(highest_price, atr_val, trailing_atr_mult)

            exit_reasons = []
            if signal == -1:
                exit_reasons.append('signal')
            if price <= hard_stop:
                exit_reasons.append('hard_stop')
            if price <= trailing_stop:
                exit_reasons.append('trailing_stop')
            if trading_halted:
                exit_reasons.append('daily_limit')
            if i == len(df) - 1:
                exit_reasons.append('end_of_data')

            if exit_reasons:
                exit_price = price * (1 - slippage)
                gross_pnl = (exit_price - entry_price) * shares
                fees = (entry_price * brokerage + exit_price * brokerage) * shares
                net_pnl = gross_pnl - fees

                equity += net_pnl
                daily_pnl += net_pnl
                peak = max(peak, equity)

                trades.append({
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'shares': shares,
                    'pnl': round(net_pnl, 2),
                    'exit_reason': exit_reasons[0],
                    'win': net_pnl > 0,
                })
                position = 0

        # ── ENTRY LOGIC ────────────────────────────────────────────────────
        elif position == 0 and signal == 1 and not trading_halted:
            entry_price = price * (1 + slippage)
            shares = atr_position_size(
                equity=equity,
                current_price=entry_price,
                atr_value=atr_val,
                risk_per_trade_pct=risk_per_trade,
                atr_multiplier=atr_stop_mult,
            )

            if shares > 0:
                hard_stop = compute_atr_stop(entry_price, atr_val, atr_stop_mult, 'long')
                highest_price = price
                position = 1

        # ── Record bar state ───────────────────────────────────────────────
        equity_curve.append(equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        drawdowns.append(dd)

    return trades, equity_curve, drawdowns