"""
main.py — End-to-end pipeline runner

Run backtest:
    python main.py --mode backtest --ticker RELIANCE.NS --interval 5m

Run live trading:
    python main.py --mode live --symbol RELIANCE --dry-run
"""

import time
from src.data_fatch   import fetch_candles
from src.indicators   import add_all_indicators
from src.strategies   import generate_signals
from src.backtest     import backtest
from src.performance  import performance_metrics, print_metrics
from src.visualizer   import plot_results


# ══════════════════════════════════════════════════════════════════════════════
# Step progress printer
# ══════════════════════════════════════════════════════════════════════════════

def _step(n, total, label):
    bar_filled = int(n / total * 20)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"\n  \033[36m[{n}/{total}]\033[0m  \033[1m{label}\033[0m")
    print(f"         \033[36m{bar}\033[0m  {int(n/total*100)}%")

def _done(label):
    print(f"         \033[32m✔  {label}\033[0m")

def _divider():
    print("\033[2m" + "  " + "─" * 58 + "\033[0m")


# ══════════════════════════════════════════════════════════════════════════════
# Backtest pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(ticker, start, end, interval, capital):
    
    print("\n\033[1m\033[36m" + "=" * 60 + "\033[0m")
    print(f"  \033[1m\033[97m  ALGO TRADING BACKTEST ENGINE\033[0m")
    print(f"  \033[2m  Ticker: {ticker}  |  Interval: {interval}  |  Capital: Rs.{capital:,.0f}\033[0m")
    print("\033[1m\033[36m" + "=" * 60 + "\033[0m")

    t0 = time.time()

    # ── Step 1 — Data ──────────────────────────────────────────────────────
    _step(1, 5, "Fetching market data")
    df = fetch_candles()
    _done(f"Loaded {len(df):,} bars  ({df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]})")

    # ── Step 2 — Indicators ────────────────────────────────────────────────
    _step(2, 5, "Computing technical indicators")
    df = add_all_indicators(df)
    _done("EMA9/21/50  •  RSI  •  MACD  •  Bollinger Bands  •  ATR")

    # ── Step 3 — Signals ───────────────────────────────────────────────────
    _step(3, 5, "Generating ML signals  (walk-forward validation)")
    _divider()
    df = generate_signals(
        df,
        lookahead=10,
        profit_target=0.005,
        confidence_threshold=0.55,
        n_splits=5,
        model_type='rf',
    )
    _divider()
    buy_sig  = int((df['signal'] == 1).sum())
    sell_sig = int((df['signal'] == -1).sum())
    _done(f"Buy signals: {buy_sig}   Sell signals: {sell_sig}")

    # ── Step 4 — Backtest ──────────────────────────────────────────────────
    _step(4, 5, "Running event-driven backtest")
    trades, equity_curve, drawdowns = backtest(
        df,
        capital=capital,
        brokerage=0.0005,
        slippage=0.0005,
        risk_per_trade=0.01,
        atr_stop_mult=2.0,
        trailing_atr_mult=1.5,
    )
    _done(f"Simulated {len(trades)} trades  |  "
          f"Final equity: Rs.{equity_curve[-1]:,.2f}" if equity_curve else "No equity data")

    # ── Step 5 — Metrics & Display ─────────────────────────────────────────
    _step(5, 5, "Computing performance metrics")
    metrics = performance_metrics(equity_curve, trades)
    _done(f"Elapsed: {time.time() - t0:.1f}s")

    # Decorated terminal report
    print_metrics(metrics, ticker=ticker)

    # ── Visualisation ──────────────────────────────────────────────────────
    print("  \033[36mGenerating visualisation dashboard...\033[0m")
    try:
        plot_results(
            df,
            trades,
            equity_curve,
            drawdowns,
            ticker=ticker,
            save_path=f"backtest_{ticker.replace('.', '_')}.png",
            show=True,          # flip to False in headless environments
        )
    except Exception as e:
        print(f"  \033[33m⚠  Visualisation skipped: {e}\033[0m")

    return df, trades, equity_curve


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_backtest('TCS', '2024-01-01', '2025-01-01', '5min', capital=100_000)