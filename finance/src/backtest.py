def backtest(df, capital=100000, brokerage=0.0005, slippage=0.0005, risk_per_trade=0.01, stop_loss_pct=0.005):
    equity_curve = [capital]
    trades = []
    drawdowns = [0]
    position = 0
    entry_price = 0
    peak = capital
    
    for i in range(1, len(df)):
        if df['signal'].iloc[i] == 1 and position == 0:
            position = 1
            entry_price = df['close'].iloc[i] * (1 + slippage)
        elif df['signal'].iloc[i] == -1 and position == 1:
            exit_price = df['close'].iloc[i] * (1 - slippage)
            pnl = (exit_price - entry_price) - (entry_price * brokerage) - (exit_price * brokerage)
            trades.append(pnl)
            equity_curve.append(equity_curve[-1] + pnl)
            peak = max(peak, equity_curve[-1])
            drawdowns.append((equity_curve[-1] - peak) / peak)
            position = 0
        elif position == 1:
            current_loss = (df['close'].iloc[i] - entry_price) / entry_price
            if current_loss <= -stop_loss_pct:
                exit_price = df['close'].iloc[i] * (1 - slippage)
                pnl = (exit_price - entry_price) - (entry_price * brokerage) - (exit_price * brokerage)
                trades.append(pnl)
                equity_curve.append(equity_curve[-1] + pnl)
                peak = max(peak, equity_curve[-1])
                drawdowns.append((equity_curve[-1] - peak) / peak)
                position = 0
    
    return trades, equity_curve, drawdowns
