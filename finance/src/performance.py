def performance_metrics(equity_curve):
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    return {
        "Total Return %": round(total_return * 100, 2),
        "Final Equity": round(equity_curve[-1], 2)
    }
