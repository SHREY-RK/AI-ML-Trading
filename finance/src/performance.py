"""
performance.py — Strategy performance analytics

Metrics:
  • Total return %
  • CAGR (annualised return)
  • Sharpe ratio     — return per unit of total risk
  • Sortino ratio    — return per unit of downside risk
  • Max drawdown %
  • Win rate %
  • Profit factor    — gross profit / gross loss
  • Avg win / avg loss / expectancy per trade
  • Total trades
"""

import numpy as np


def performance_metrics(equity_curve, trades, bars_per_year=252 * 78):
    """
    Parameters
    ----------
    equity_curve  : list of equity values (one per bar)
    trades        : list of trade dicts returned by backtest()
    bars_per_year : number of bars in a trading year
                    252 days × 78 bars/day for 5-minute NSE data
                    Use 252 for daily data.

    Returns
    -------
    dict of metrics, ready to print or log.
    """
    if len(equity_curve) < 2:
        return {"error": "Not enough data"}

    eq = np.array(equity_curve, dtype=float)
    initial = eq[0]
    final = eq[-1]
    n_bars = len(eq)

    # ── Returns ────────────────────────────────────────────────────────────
    total_return_pct = (final - initial) / initial * 100
    years = n_bars / bars_per_year
    cagr_pct = ((final / initial) ** (1 / max(years, 1e-9)) - 1) * 100 if years > 0 else 0.0

    bar_returns = np.diff(eq) / eq[:-1]          # per-bar % returns

    # ── Sharpe Ratio ───────────────────────────────────────────────────────
    # Risk-free rate ≈ 0 for intraday; annualised using sqrt(bars_per_year)
    sharpe = 0.0
    if bar_returns.std() > 0:
        sharpe = (bar_returns.mean() / bar_returns.std()) * np.sqrt(bars_per_year)

    # ── Sortino Ratio ──────────────────────────────────────────────────────
    downside = bar_returns[bar_returns < 0]
    sortino = 0.0
    if len(downside) > 0 and downside.std() > 0:
        sortino = (bar_returns.mean() / downside.std()) * np.sqrt(bars_per_year)

    # ── Max Drawdown ───────────────────────────────────────────────────────
    running_peak = np.maximum.accumulate(eq)
    drawdowns = (eq - running_peak) / running_peak
    max_drawdown_pct = drawdowns.min() * 100

    # ── Trade statistics ───────────────────────────────────────────────────
    if not trades:
        win_rate = profit_factor = avg_win = avg_loss = expectancy = 0.0
        n_trades = 0
    else:
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        n_trades = len(pnls)
        win_rate = len(wins) / n_trades * 100 if n_trades else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy: average ₹ earned per trade
        expectancy = np.mean(pnls) if pnls else 0.0

    # ── Exit reason breakdown ──────────────────────────────────────────────
    exit_reasons = {}
    for t in trades:
        reason = t.get('exit_reason', 'unknown')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        "Initial Capital":      f"₹{initial:,.0f}",
        "Final Equity":         f"₹{final:,.2f}",
        "Total Return %":       f"{total_return_pct:.2f}%",
        "CAGR %":               f"{cagr_pct:.2f}%",
        "Sharpe Ratio":         f"{sharpe:.2f}",
        "Sortino Ratio":        f"{sortino:.2f}",
        "Max Drawdown %":       f"{max_drawdown_pct:.2f}%",
        "Total Trades":         n_trades,
        "Win Rate %":           f"{win_rate:.1f}%",
        "Profit Factor":        f"{profit_factor:.2f}",
        "Avg Win (₹)":          f"{avg_win:.2f}",
        "Avg Loss (₹)":         f"{avg_loss:.2f}",
        "Expectancy per Trade": f"₹{expectancy:.2f}",
        "Exit Reasons":         exit_reasons,
    }


def print_metrics(metrics):
    """Pretty-prints the metrics dict."""
    print("\n" + "=" * 45)
    print("  📊  BACKTEST PERFORMANCE REPORT")
    print("=" * 45)
    for key, val in metrics.items():
        if key == "Exit Reasons":
            print(f"\n  {'Exit breakdown':30}")
            for reason, count in val.items():
                print(f"    {reason:<28} {count}")
        else:
            print(f"  {key:<30} {val}")
    print("=" * 45 + "\n")