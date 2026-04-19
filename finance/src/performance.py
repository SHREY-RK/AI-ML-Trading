"""
performance.py — Strategy performance analytics + rich terminal output

Metrics computed:
  • Total return %          • CAGR
  • Sharpe ratio            • Sortino ratio
  • Max drawdown %          • Win rate %
  • Profit factor           • Avg win / avg loss
  • Expectancy per trade    • Total trades

Terminal output uses ANSI colours — no extra dependencies needed.
"""

import numpy as np
import sys
import re


# ══════════════════════════════════════════════════════════════════════════════
# ANSI colour helpers
# ══════════════════════════════════════════════════════════════════════════════

_USE_COLOR = True   # set False for plain log files

def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else str(text)

def green(t):   return _c(t, "32")
def red(t):     return _c(t, "31")
def yellow(t):  return _c(t, "33")
def cyan(t):    return _c(t, "36")
def bold(t):    return _c(t, "1")
def dim(t):     return _c(t, "2")
def white(t):   return _c(t, "97")

def _strip_ansi(s):
    return re.sub(r'\033\[[0-9;]*m', '', str(s))


# ══════════════════════════════════════════════════════════════════════════════
# Core metric calculation
# ══════════════════════════════════════════════════════════════════════════════

def performance_metrics(equity_curve, trades, bars_per_year=252 * 78):
    """
    Parameters
    ----------
    equity_curve  : list of equity values (one per bar)
    trades        : list of trade dicts returned by backtest()
    bars_per_year : bars in a trading year (252 x 78 for 5-min NSE)

    Returns
    -------
    dict with raw numeric (_prefixed) and formatted string values
    """
    if len(equity_curve) < 2:
        return {"error": "Not enough data"}

    eq      = np.array(equity_curve, dtype=float)
    initial = eq[0]
    final   = eq[-1]
    n_bars  = len(eq)

    total_return_pct = (final - initial) / initial * 100
    years   = n_bars / bars_per_year
    cagr_pct = ((final / initial) ** (1 / max(years, 1e-9)) - 1) * 100

    bar_returns = np.diff(eq) / eq[:-1]

    sharpe = sortino = 0.0
    if bar_returns.std() > 0:
        sharpe = (bar_returns.mean() / bar_returns.std()) * np.sqrt(bars_per_year)
    downside = bar_returns[bar_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (bar_returns.mean() / downside.std()) * np.sqrt(bars_per_year)

    running_peak     = np.maximum.accumulate(eq)
    drawdown_series  = (eq - running_peak) / running_peak
    max_drawdown_pct = drawdown_series.min() * 100

    if not trades:
        win_rate = profit_factor = avg_win = avg_loss = expectancy = 0.0
        n_trades = 0
    else:
        pnls   = [t['pnl'] for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        n_trades      = len(pnls)
        win_rate      = len(wins) / n_trades * 100
        avg_win       = float(np.mean(wins))   if wins   else 0.0
        avg_loss      = float(np.mean(losses)) if losses else 0.0
        gross_profit  = sum(wins)
        gross_loss    = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        expectancy    = float(np.mean(pnls))

    exit_reasons: dict = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return dict(
        _initial=initial, _final=final,
        _total_return_pct=total_return_pct, _cagr_pct=cagr_pct,
        _sharpe=sharpe, _sortino=sortino,
        _max_drawdown_pct=max_drawdown_pct,
        _n_trades=n_trades, _win_rate=win_rate,
        _profit_factor=profit_factor,
        _avg_win=avg_win, _avg_loss=avg_loss,
        _expectancy=expectancy, _exit_reasons=exit_reasons,
        **{
            "Initial Capital":      f"Rs.{initial:,.0f}",
            "Final Equity":         f"Rs.{final:,.2f}",
            "Total Return %":       f"{total_return_pct:.2f}%",
            "CAGR %":               f"{cagr_pct:.2f}%",
            "Sharpe Ratio":         f"{sharpe:.2f}",
            "Sortino Ratio":        f"{sortino:.2f}",
            "Max Drawdown %":       f"{max_drawdown_pct:.2f}%",
            "Total Trades":         n_trades,
            "Win Rate %":           f"{win_rate:.1f}%",
            "Profit Factor":        f"{profit_factor:.2f}",
            "Avg Win (Rs.)":        f"Rs.{avg_win:.2f}",
            "Avg Loss (Rs.)":       f"Rs.{avg_loss:.2f}",
            "Expectancy per Trade": f"Rs.{expectancy:.2f}",
            "Exit Reasons":         exit_reasons,
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# Colour-graded interpretation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _grade_sharpe(v):
    if v >= 2.0:   return green("Excellent -- very strong risk-adjusted return")
    if v >= 1.0:   return green("Good -- solid risk-adjusted return")
    if v >= 0.5:   return yellow("Acceptable -- room to improve")
    if v >= 0.0:   return yellow("Weak -- barely beats risk taken")
    return red("Negative -- destroys value vs holding cash")

def _grade_drawdown(v):
    if v >= -5:    return green(f"{v:.1f}%  -- very low, great capital preservation")
    if v >= -15:   return yellow(f"{v:.1f}%  -- moderate, manageable with discipline")
    if v >= -25:   return yellow(f"{v:.1f}%  -- high, tighten stop-losses")
    return red(f"{v:.1f}%  -- severe, may be over-leveraged")

def _grade_win_rate(v):
    if v >= 60:    return green(f"{v:.1f}%  -- strong win rate")
    if v >= 50:    return yellow(f"{v:.1f}%  -- average, relies on large wins")
    return red(f"{v:.1f}%  -- below 50%, profit factor must compensate")

def _grade_profit_factor(v):
    if v >= 2.0:   return green(f"{v:.2f}  -- earns Rs.{v:.2f} per Re.1 lost")
    if v >= 1.5:   return green(f"{v:.2f}  -- profitable edge")
    if v >= 1.0:   return yellow(f"{v:.2f}  -- marginal edge, barely profitable")
    return red(f"{v:.2f}  -- losing strategy overall")

def _grade_expectancy(v):
    if v > 0:      return green(f"Rs.{v:.2f}  -- earn this per trade on average")
    return red(f"Rs.{v:.2f}  -- losing money on average per trade")

def _grade_return(v):
    if v > 20:     return green(f"{v:.2f}%")
    if v > 0:      return yellow(f"{v:.2f}%")
    return red(f"{v:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Box-drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

W = 74

def _top():   return cyan(bold("╔" + "═" * (W - 2) + "╗"))
def _mid():   return cyan(bold("╠" + "═" * (W - 2) + "╣"))
def _bot():   return cyan(bold("╚" + "═" * (W - 2) + "╝"))
def _blank(): return cyan(bold("║" + " " * (W - 2) + "║"))

def _section(title):
    cl = len(_strip_ansi(title))
    pad = W - 2 - cl
    lp, rp = pad // 2, pad - pad // 2
    return cyan(bold(f"║{' ' * lp}{title}{' ' * rp}║"))

def _row(label, value, label_w=28):
    content = f"  {label:<{label_w}} {value}"
    vis = len(_strip_ansi(content))
    pad = W - 2 - vis
    return cyan(bold(f"║{content}{' ' * max(pad, 0)}║"))

def _text_row(text):
    content = f"  {text}"
    vis = len(_strip_ansi(content))
    pad = W - 2 - vis
    return cyan(bold(f"║{content}{' ' * max(pad, 0)}║"))


# ══════════════════════════════════════════════════════════════════════════════
# Plain-English Summary
# ══════════════════════════════════════════════════════════════════════════════

def _plain_english_summary(m):
    lines = []
    ret = m['_total_return_pct']
    ret_s = green(f"+{ret:.1f}%") if ret > 0 else red(f"{ret:.1f}%")
    lines.append(f"Rs.{m['_initial']:,.0f} grew to Rs.{m['_final']:,.0f}  ({ret_s} total return)")

    if m['_n_trades'] == 0:
        lines.append(red("No trades generated. Confidence threshold may be too high."))
        return lines

    if m['_win_rate'] >= 50:
        lines.append(green(f"{m['_win_rate']:.0f}% of trades won -- more than half made money."))
    else:
        lines.append(yellow(f"Only {m['_win_rate']:.0f}% win rate -- needs large wins to profit."))

    pf = m['_profit_factor']
    if pf >= 1.0:
        lines.append(green(f"Profit factor {pf:.2f}: earns Rs.{pf:.2f} per Re.1 lost."))
    else:
        lines.append(red(f"Profit factor {pf:.2f}: loses more than it earns over time."))

    dd = m['_max_drawdown_pct']
    if dd < -20:
        lines.append(red(f"Peak-to-trough loss was {dd:.1f}%. Consider smaller positions."))
    else:
        lines.append(green(f"Max drawdown only {dd:.1f}% -- capital well protected."))

    sh = m['_sharpe']
    if sh >= 1.0:
        lines.append(green(f"Sharpe {sh:.2f} -- solid return per unit of risk."))
    elif sh >= 0:
        lines.append(yellow(f"Sharpe {sh:.2f} -- marginal risk-adjusted return."))
    else:
        lines.append(red(f"Sharpe {sh:.2f} -- too much risk for the return generated."))

    exp = m['_expectancy']
    if exp > 0:
        lines.append(green(f"Average of Rs.{exp:.2f} earned per trade. Keep running this edge."))
    else:
        lines.append(red(f"Average loss of Rs.{abs(exp):.2f} per trade. Edge needs improvement."))

    exit_reasons = m.get('_exit_reasons', {})
    if exit_reasons:
        top_r = max(exit_reasons, key=exit_reasons.get)
        top_c = exit_reasons[top_r]
        pct   = top_c / m['_n_trades'] * 100
        tips = {
            "signal":        green(f"Model exit signals drive {pct:.0f}% of closes -- healthy."),
            "hard_stop":     yellow(f"Hard stop fires on {pct:.0f}% of trades -- try wider stops."),
            "trailing_stop": green(f"Trailing stop secures profit on {pct:.0f}% of trades."),
            "daily_limit":   red(f"Daily loss limit fires on {pct:.0f}% of trades -- reduce size."),
            "end_of_data":   dim(f"{pct:.0f}% hit end of dataset -- extend the data window."),
        }
        tip = tips.get(top_r)
        if tip:
            lines.append(f"Top exit '{top_r}': {tip}")

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Public print_metrics
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(metrics, ticker="Strategy"):
    """
    Prints a fully decorated, colour-coded backtest report to the terminal.
    """
    if "error" in metrics:
        print(red(f"\n  ERROR: {metrics['error']}"))
        return

    m = metrics
    out = ["\n", _top(), _blank()]

    # Title
    title = bold(white(f"  BACKTEST REPORT  --  {ticker}  |  {m['_n_trades']} trades  "))
    out.append(_section(title))
    out.append(_blank())

    # Capital
    out.append(_mid())
    out.append(_section(yellow("  CAPITAL  ")))
    out.append(_mid())
    out.append(_row("Starting Capital",   white(f"Rs.{m['_initial']:>12,.0f}")))
    out.append(_row("Final Equity",       white(f"Rs.{m['_final']:>12,.2f}")))
    out.append(_row("Total Return",       _grade_return(m['_total_return_pct'])))
    out.append(_row("CAGR (annualised)",  _grade_return(m['_cagr_pct'])))

    # Risk metrics
    out.append(_mid())
    out.append(_section(yellow("  RISK-ADJUSTED PERFORMANCE  ")))
    out.append(_mid())
    out.append(_row("Sharpe Ratio",  _grade_sharpe(m['_sharpe'])))
    so_col = green if m['_sortino'] >= 1 else yellow
    out.append(_row("Sortino Ratio", so_col(f"{m['_sortino']:.2f}")))
    out.append(_row("Max Drawdown",  _grade_drawdown(m['_max_drawdown_pct'])))

    # Trade stats
    out.append(_mid())
    out.append(_section(yellow("  TRADE STATISTICS  ")))
    out.append(_mid())
    out.append(_row("Total Trades",        white(str(m['_n_trades']))))
    out.append(_row("Win Rate",            _grade_win_rate(m['_win_rate'])))
    out.append(_row("Profit Factor",       _grade_profit_factor(m['_profit_factor'])))
    out.append(_row("Avg Winning Trade",   green(f"Rs.{m['_avg_win']:.2f}")))
    out.append(_row("Avg Losing Trade",    red(f"Rs.{m['_avg_loss']:.2f}")))
    out.append(_row("Expectancy / Trade",  _grade_expectancy(m['_expectancy'])))

    # Exit reasons
    exit_reasons = m.get('_exit_reasons', {})
    if exit_reasons:
        out.append(_mid())
        out.append(_section(yellow("  HOW TRADES ENDED  ")))
        out.append(_mid())
        total = m['_n_trades'] or 1
        icons = {"signal": "[SIG]", "hard_stop": "[STP]",
                 "trailing_stop": "[TRL]", "daily_limit": "[LMT]",
                 "end_of_data": "[END]", "unknown": "[???]"}
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            icon   = icons.get(reason, "[ ] ")
            pct    = count / total * 100
            filled = int(pct / 5)
            bar    = cyan("█" * filled) + dim("░" * (20 - filled))
            label  = f"{icon}  {reason}"
            val    = f"{bar}  {white(str(count))}  {dim(f'{pct:.0f}%')}"
            out.append(_row(label, val, label_w=22))

    # Plain-English summary
    out.append(_mid())
    out.append(_section(yellow("  PLAIN-ENGLISH SUMMARY  ")))
    out.append(_mid())
    for s in _plain_english_summary(m):
        out.append(_text_row(s))

    out.append(_blank())
    out.append(_bot())
    out.append("")
    print("\n".join(out))