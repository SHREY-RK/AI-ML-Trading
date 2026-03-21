from src.login import generate_session
from src.data_fatch import fetch_candles
from src.indicators import ema, rsi, volatility
from src.strategies import generate_signals
from src.backtest import backtest
from src.performance import performance_metrics

if __name__ == "__main__":
    api = generate_session()
    if not api:
        exit(1)
    
    df = fetch_candles(api, token="11536", fromdate="2025-11-01 09:15", todate="2026-02-20 15:30")
    
    df["ema9"] = ema(df, 9)
    df["ema21"] = ema(df, 21)
    df["rsi"] = rsi(df)
    df["volatility"] = volatility(df)
    df.dropna(inplace=True)
    
    df = generate_signals(df)
    
    trades, equity_curve, drawdowns = backtest(df, capital=100000, brokerage=0.0005, slippage=0.0005, risk_per_trade=0.01, stop_loss_pct=0.005)
    
    metrics = performance_metrics(equity_curve)
    
    print("\n========== BACKTEST RESULTS ==========")
    print(f"Total Trades: {len(trades)}")
    print(f"Winning Trades: {sum(1 for t in trades if t > 0)}")
    print(f"Losing Trades: {sum(1 for t in trades if t < 0)}")
    print(f"Win Rate: {round(sum(1 for t in trades if t > 0) / len(trades) * 100, 2) if trades else 0}%")
    print(f"Max Drawdown: {round(min(drawdowns) * 100, 2)}%")
    print(f"Total Return: {metrics['Total Return %']}%")
    print(f"Final Equity: ₹{metrics['Final Equity']}")
    print("======================================\n")