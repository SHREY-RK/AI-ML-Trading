"""
main.py — End-to-end pipeline runner

Run backtest:
    python main.py --mode backtest --ticker RELIANCE.NS --interval 5m

Run live trading:
    python main.py --mode live --symbol RELIANCE --dry-run
"""

import argparse
from src.data_fatch import fetch_candles
from src.indicators import add_all_indicators
from src.strategies import generate_signals
from src.backtest import backtest
from src.performance import performance_metrics, print_metrics


stock_file = "WIPRO_5minute.csv"
def run_backtest(ticker, start, end, interval, capital):
    print(f"\n{'='*50}")
    print(f"  Backtest: {ticker} | {interval} | {start} → {end}")
    print(f"{'='*50}")


    # 1. Fetch data
    df = fetch_candles(stock_file)
    print(f"  Loaded {len(df)} bars.")

    # 2. Add indicators
    df = add_all_indicators(df)

    # 3. Generate ML signals (walk-forward)
    df = generate_signals(
        df,
        lookahead=8,  # how many bars ahead to look for the profit target
        profit_target=0.005,  # 0.5% profit target for labeling the data
        confidence_threshold=0.65,  # only take signals where the model is at least 55% confident
        n_splits=10,  # number of walk-forward splits for training/testing the model
        model_type='rf',  # 'rf' for Random Forest, 'gb' for Gradient Boosting
    )

    # 4. Run backtest
    trades, equity_curve, drawdowns = backtest(
        df,
        capital=capital,
        brokerage=0.0005,  # 0.05% per side
        slippage=0.0005,   # 0.05% per side
        risk_per_trade=0.01,  # risk 1% of equity per trade
        atr_stop_mult=2.0,   # hard stop = 2× ATR below entry
        trailing_atr_mult=1.5,   # trailing stop = 1.5× ATR below highest
    )

    # 5. Print performance
    metrics = performance_metrics(equity_curve, trades)
    print_metrics(metrics)

    return df, trades, equity_curve


# def run_live(symbol, dry_run):
#     from src.login import generate_session
#     from src.execution import Executor
#     from datetime import datetime, timedelta

#     api = generate_session()
#     if not api:
#         print("Login failed. Exiting.")
#         return

#     executor = Executor(api, symbol=symbol, dry_run=dry_run)

#     # Pull recent data for signal generation
#     from download_data import download_massive_data
#     download_massive_data(api, symbol, days_back=30, interval=5)

#     import pandas as pd
#     df = pd.read_csv(f"{symbol}_massive_5m.csv", parse_dates=['timestamp'])
#     df = add_all_indicators(df)
#     df = generate_signals(df)

#     # The last row's signal is our current signal
#     last_signal = df['signal'].iloc[-1]
#     ltp = executor.get_ltp()
#     print(f"\nLatest signal: {last_signal} | LTP: ₹{ltp}")

#     if last_signal == 1 and executor.position == 0:
#         atr_val = df['atr'].iloc[-1]
#         from risk import atr_position_size
#         qty = atr_position_size(100000, ltp, atr_val)
#         executor.buy(quantity=qty)

#     elif last_signal == -1 and executor.position > 0:
#         executor.square_off()

#     executor.print_session_summary()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest')
    # parser.add_argument('--ticker', default='RELIANCE.NS')
    # parser.add_argument('--symbol', default='RELIANCE')
    # parser.add_argument('--start', default='2015-01-01')
    # parser.add_argument('--end', default='2026-01-01')
    # parser.add_argument('--interval', default='1d')
    # parser.add_argument('--capital', type=float, default=100000)
    # parser.add_argument('--dry-run', action='store_true')
    # args = parser.parse_args()

    # if args.mode == 'backtest':
    run_backtest(stock_file, '2020-03-16', '2020-03-27', '1min', capital=100000)
    
    
    # else:
    #     run_live(args.symbol, args.dry_run)