"""
backtest.py — Event-driven backtest engine (Long + Short)

Supports two trade directions:
  • LONG  (position =  1) — buy first, sell higher → profit when price rises
  • SHORT (position = -1) — sell first, buy lower  → profit when price falls

Signal conventions
------------------
  signal =  1  →  enter LONG  (or exit an open SHORT)
  signal = -1  →  enter SHORT (or exit an open LONG)
  signal =  0  →  hold

Stop logic
----------
  Long  : hard stop BELOW entry; trailing stop rises with highest_price
  Short : hard stop ABOVE entry; trailing stop falls  with lowest_price

Risk controls (unchanged)
-----------
  • ATR-based position sizing (risk_per_trade % of equity per trade)
  • Daily loss circuit breaker (max_daily_loss_pct)
"""

import numpy as np
from src.risk import (
    atr_position_size,
    compute_atr_stop,
    compute_trailing_stop,
    compute_short_trailing_stop,
    check_daily_loss_limit,
)


def backtest(
    df,
    capital=100000,
    brokerage=0.0005,        # 0.05 % per leg (realistic for NSE intraday)
    slippage=0.0005,         # 0.05 % per leg
    risk_per_trade=0.01,     # risk 1 % of equity per trade
    atr_stop_mult=2.0,       # hard stop = 2 × ATR away from entry
    trailing_atr_mult=1.5,   # trailing stop = 1.5 × ATR from the best price seen
    max_daily_loss_pct=0.02,
    allow_short=True,        # set False to disable short selling
):
    """
    Simulates the strategy on out-of-sample signal data.

    Returns
    -------
    trades       : list of dicts with entry/exit details
    equity_curve : list of equity values, one per bar (length == len(df))
    drawdowns    : list of drawdown values, one per bar
    """
    equity = capital
    position = 0            # 0 = flat | 1 = long | -1 = short
    entry_price = 0.0
    shares = 0
    direction = None        # 'long' or 'short'

    # Long tracking
    highest_price = 0.0     # highest price seen while in a long trade
    # Short tracking
    lowest_price = float('inf')   # lowest price seen while in a short trade

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
        atr_val = row.get('atr', price * 0.005)   # fallback: 0.5 % of price

        # ── Daily reset ────────────────────────────────────────────────────
        bar_day = row['timestamp'].date() if hasattr(row['timestamp'], 'date') else None
        if bar_day != current_day:
            daily_pnl = 0.0
            current_day = bar_day

        # ── Circuit breaker ────────────────────────────────────────────────
        trading_halted = check_daily_loss_limit(daily_pnl, equity, max_daily_loss_pct)

        # ══════════════════════════════════════════════════════════════════
        # EXIT LOGIC — LONG position
        # ══════════════════════════════════════════════════════════════════
        if position == 1:
            # Keep updating the best price (used by trailing stop)
            if price > highest_price:
                highest_price = price

            trailing_stop = compute_trailing_stop(highest_price, atr_val, trailing_atr_mult)

            exit_reasons = []
            if signal == -1:
                exit_reasons.append('signal')          # model says go short → exit long first
            if price <= hard_stop:
                exit_reasons.append('hard_stop')
            if price <= trailing_stop:
                exit_reasons.append('trailing_stop')
            if trading_halted:
                exit_reasons.append('daily_limit')
            if i == len(df) - 1:
                exit_reasons.append('end_of_data')

            if exit_reasons:
                # Sell at a slight discount (slippage on the exit leg)
                exit_price = price * (1 - slippage)
                gross_pnl = (exit_price - entry_price) * shares
                fees = (entry_price + exit_price) * brokerage * shares
                net_pnl = gross_pnl - fees

                equity += net_pnl
                daily_pnl += net_pnl
                peak = max(peak, equity)

                trades.append({
                    'direction':   'long',
                    'entry_price': round(entry_price, 2),
                    'exit_price':  round(exit_price, 2),
                    'shares':      shares,
                    'pnl':         round(net_pnl, 2),
                    'exit_reason': exit_reasons[0],
                    'win':         net_pnl > 0,
                })
                position = 0

        # ══════════════════════════════════════════════════════════════════
        # EXIT LOGIC — SHORT position
        # ══════════════════════════════════════════════════════════════════
        elif position == -1:
            # Keep updating the best price (lowest seen → trailing stop tightens)
            if price < lowest_price:
                lowest_price = price

            trailing_stop = compute_short_trailing_stop(lowest_price, atr_val, trailing_atr_mult)

            exit_reasons = []
            if signal == 1:
                exit_reasons.append('signal')          # model says go long → exit short first
            if price >= hard_stop:
                exit_reasons.append('hard_stop')       # price moved ABOVE our hard stop
            if price >= trailing_stop:
                exit_reasons.append('trailing_stop')   # trailing stop breached from below
            if trading_halted:
                exit_reasons.append('daily_limit')
            if i == len(df) - 1:
                exit_reasons.append('end_of_data')

            if exit_reasons:
                # Buy-to-cover at a slight premium (slippage on the exit leg)
                exit_price = price * (1 + slippage)
                # Short PnL: we sold high (entry) and buy back low (exit)
                gross_pnl = (entry_price - exit_price) * shares
                fees = (entry_price + exit_price) * brokerage * shares
                net_pnl = gross_pnl - fees

                equity += net_pnl
                daily_pnl += net_pnl
                peak = max(peak, equity)

                trades.append({
                    'direction':   'short',
                    'entry_price': round(entry_price, 2),
                    'exit_price':  round(exit_price, 2),
                    'shares':      shares,
                    'pnl':         round(net_pnl, 2),
                    'exit_reason': exit_reasons[0],
                    'win':         net_pnl > 0,
                })
                position = 0

        # ══════════════════════════════════════════════════════════════════
        # ENTRY LOGIC — only when flat (position == 0)
        # ══════════════════════════════════════════════════════════════════
        if position == 0 and not trading_halted:

            # ── Enter LONG ─────────────────────────────────────────────────
            if signal == 1:
                entry_price = price * (1 + slippage)   # buy at slight premium
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

            # ── Enter SHORT ────────────────────────────────────────────────
            elif signal == -1 and allow_short:
                entry_price = price * (1 - slippage)   # sell short at slight discount
                shares = atr_position_size(
                    equity=equity,
                    current_price=entry_price,
                    atr_value=atr_val,
                    risk_per_trade_pct=risk_per_trade,
                    atr_multiplier=atr_stop_mult,
                )
                if shares > 0:
                    hard_stop = compute_atr_stop(entry_price, atr_val, atr_stop_mult, 'short')
                    lowest_price = price
                    position = -1

        # ── Record bar state ───────────────────────────────────────────────
        equity_curve.append(equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        drawdowns.append(dd)

    return trades, equity_curve, drawdowns