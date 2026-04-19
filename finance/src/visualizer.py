"""
visualizer.py — Backtest dashboard

Produces a single figure with 4 panels:
  1. Equity curve  (with buy/sell markers)
  2. Underwater drawdown chart
  3. Per-trade P&L waterfall  (green = win, red = loss)
  4. Exit-reason pie chart

Usage:
    from visualizer import plot_results
    plot_results(df, trades, equity_curve, drawdowns, ticker="RELIANCE")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


# ── Color palette ──────────────────────────────────────────────────────────
C_BG        = "#0f1117"
C_PANEL     = "#1a1d27"
C_TEXT      = "#e8eaf0"
C_MUTED     = "#7f8599"
C_GREEN     = "#26c281"
C_RED       = "#e74c3c"
C_BLUE      = "#4a9eff"
C_YELLOW    = "#f0c040"
C_ORANGE    = "#f39c12"
C_GRID      = "#2a2d3a"

EXIT_COLORS = {
    "signal":        C_BLUE,
    "hard_stop":     C_RED,
    "trailing_stop": C_ORANGE,
    "daily_limit":   C_YELLOW,
    "end_of_data":   C_MUTED,
    "unknown":       C_MUTED,
}


def _rupee(val, pos=None):
    """Axis formatter: ₹1,23,456"""
    if abs(val) >= 1_000:
        return f"₹{val:,.0f}"
    return f"₹{val:.0f}"


def _pct(val, pos=None):
    return f"{val:.1f}%"


def _setup_axes(ax, title=""):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.grid(color=C_GRID, linewidth=0.5, linestyle='--', alpha=0.6)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight='bold', pad=8)


# ── Panel 1 — Equity Curve ─────────────────────────────────────────────────

def _plot_equity(ax, equity_curve, trades, initial_capital):
    eq = np.array(equity_curve, dtype=float)
    xs  = np.arange(len(eq))

    # Shade above/below starting capital
    ax.fill_between(xs, initial_capital, eq,
                    where=eq >= initial_capital,
                    alpha=0.15, color=C_GREEN, interpolate=True)
    ax.fill_between(xs, initial_capital, eq,
                    where=eq < initial_capital,
                    alpha=0.20, color=C_RED, interpolate=True)
    ax.plot(xs, eq, color=C_BLUE, linewidth=1.5, label="Equity")
    ax.axhline(initial_capital, color=C_MUTED, linewidth=0.8,
               linestyle='--', label="Start capital")

    # Overlay win/loss markers using trade bar indices (approximate)
    # We'll just mark winning vs losing trades along the curve if we can
    ax.yaxis.set_major_formatter(FuncFormatter(_rupee))
    ax.set_xlabel("Bar index")
    ax.legend(loc='upper left', fontsize=8,
              facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
    _setup_axes(ax, "📈  Equity Curve")


# ── Panel 2 — Drawdown ─────────────────────────────────────────────────────

def _plot_drawdown(ax, drawdowns):
    dd = np.array(drawdowns, dtype=float) * 100
    xs  = np.arange(len(dd))

    ax.fill_between(xs, dd, 0, alpha=0.5, color=C_RED)
    ax.plot(xs, dd, color=C_RED, linewidth=1.0)
    ax.axhline(0, color=C_MUTED, linewidth=0.6)

    max_dd_idx = np.argmin(dd)
    ax.annotate(f"Max {dd[max_dd_idx]:.1f}%",
                xy=(max_dd_idx, dd[max_dd_idx]),
                xytext=(max_dd_idx + len(dd) * 0.05, dd[max_dd_idx] * 0.6),
                color=C_RED, fontsize=8,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))

    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_xlabel("Bar index")
    _setup_axes(ax, "📉  Underwater Drawdown")


# ── Panel 3 — Per-Trade P&L waterfall ──────────────────────────────────────

def _plot_trade_pnl(ax, trades):
    if not trades:
        ax.text(0.5, 0.5, "No trades recorded", color=C_MUTED,
                ha='center', va='center', transform=ax.transAxes)
        _setup_axes(ax, "💹  Per-Trade P&L")
        return

    pnls   = [t['pnl'] for t in trades]
    colors = [C_GREEN if p > 0 else C_RED for p in pnls]
    xs     = np.arange(len(pnls))

    ax.bar(xs, pnls, color=colors, width=0.7, alpha=0.85)
    ax.axhline(0, color=C_MUTED, linewidth=0.8)

    # Running cumulative P&L line
    cum = np.cumsum(pnls)
    ax2 = ax.twinx()
    ax2.plot(xs, cum, color=C_YELLOW, linewidth=1.2,
             linestyle='-', label="Cumulative P&L")
    ax2.yaxis.set_major_formatter(FuncFormatter(_rupee))
    ax2.tick_params(colors=C_MUTED, labelsize=7)
    ax2.set_facecolor(C_PANEL)
    for spine in ax2.spines.values():
        spine.set_edgecolor(C_GRID)

    ax.yaxis.set_major_formatter(FuncFormatter(_rupee))
    ax.set_xlabel("Trade #")
    _setup_axes(ax, "💹  Per-Trade P&L  (yellow = cumulative)")


# ── Panel 4 — Exit reasons pie ─────────────────────────────────────────────

def _plot_exit_reasons(ax, trades):
    if not trades:
        ax.text(0.5, 0.5, "No trades recorded", color=C_MUTED,
                ha='center', va='center', transform=ax.transAxes)
        _setup_axes(ax, "🚪  Exit Reasons")
        return

    counter = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        counter[r] = counter.get(r, 0) + 1

    labels = list(counter.keys())
    sizes  = list(counter.values())
    colors = [EXIT_COLORS.get(l, C_MUTED) for l in labels]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct='%1.0f%%',
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor=C_BG, linewidth=1.5),
    )
    for at in autotexts:
        at.set_color(C_BG)
        at.set_fontsize(9)
        at.set_fontweight('bold')

    # Custom legend
    patches = [mpatches.Patch(color=colors[i], label=f"{labels[i]}  ({sizes[i]})")
               for i in range(len(labels))]
    ax.legend(handles=patches, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.10), fontsize=8,
              facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

    ax.set_facecolor(C_BG)
    ax.set_title("🚪  Exit Reasons", color=C_TEXT,
                 fontsize=10, fontweight='bold', pad=8)


# ── Public entry point ─────────────────────────────────────────────────────

def plot_results(
    df,
    trades,
    equity_curve,
    drawdowns,
    ticker="Strategy",
    save_path=None,
    show=True,
):
    """
    Parameters
    ----------
    df           : DataFrame (used for timestamps on x-axis)
    trades       : list of trade dicts from backtest()
    equity_curve : list of per-bar equity values
    drawdowns    : list of per-bar drawdown fractions
    ticker       : label shown in the title
    save_path    : if given, saves to this .png path
    show         : if True, calls plt.show()
    """
    initial_capital = equity_curve[0] if equity_curve else 100_000

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10), facecolor=C_BG)
    fig.suptitle(
        f"  🤖  Backtest Report — {ticker}  |  {len(trades)} trades  |  "
        f"{len(equity_curve):,} bars",
        color=C_TEXT, fontsize=13, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.42,
        wspace=0.30,
        left=0.07, right=0.97,
        top=0.93, bottom=0.07,
    )

    ax1 = fig.add_subplot(gs[0, :])   # equity — full top row
    ax2 = fig.add_subplot(gs[1, 0])   # drawdown
    ax3 = fig.add_subplot(gs[1, 1])   # exit reasons pie  (swap order below)

    # Rebuild 2x2: equity curve top-left, drawdown top-right,
    # trade P&L bottom-left, exit pie bottom-right
    fig.clear()
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.50,
        wspace=0.30,
        left=0.07, right=0.97,
        top=0.93, bottom=0.06,
    )

    ax_eq  = fig.add_subplot(gs[0, :])      # row 0: full width — equity
    ax_dd  = fig.add_subplot(gs[1, :])      # row 1: full width — drawdown
    ax_pnl = fig.add_subplot(gs[2, 0])     # row 2 left — trade P&L
    ax_pie = fig.add_subplot(gs[2, 1])     # row 2 right — exit pie

    _plot_equity(ax_eq, equity_curve, trades, initial_capital)
    _plot_drawdown(ax_dd, drawdowns)
    _plot_trade_pnl(ax_pnl, trades)
    _plot_exit_reasons(ax_pie, trades)

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=C_BG, bbox_inches='tight')
        print(f"  📁  Chart saved → {save_path}")

    if show:
        plt.show()

    return fig