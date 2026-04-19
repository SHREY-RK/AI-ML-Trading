"""
risk.py — Position sizing and risk management

Two complementary approaches:
  1. Fixed fractional — risk a fixed % of equity per trade
  2. ATR-based      — size position so that 1 ATR move = N% of equity loss
"""


def atr_position_size(
    equity,
    current_price,
    atr_value,
    risk_per_trade_pct=0.01,    # risk 1% of equity per trade
    atr_multiplier=2.0,          # stop = 2× ATR below entry
):
    """
    Position size using ATR-based stop distance.

    Stop distance = atr_multiplier × ATR
    Shares = (equity × risk_pct) / stop_distance

    This ensures that if the stop is hit you lose exactly `risk_per_trade_pct`
    of your current equity — regardless of how volatile the stock is.
    """
    if atr_value <= 0 or current_price <= 0:
        return 0

    stop_distance = atr_multiplier * atr_value          # in price units
    rupees_at_risk = equity * risk_per_trade_pct        # ₹ we are willing to lose
    shares = rupees_at_risk / stop_distance

    # Never spend more than 20% of equity on a single trade
    max_shares = (equity * 0.20) / current_price
    shares = min(shares, max_shares)

    return max(int(shares), 0)


def fixed_fractional_size(
    equity,
    current_price,
    fraction=0.02,          # deploy 2% of equity per trade
    max_fraction=0.20,      # hard cap at 20% per trade
):
    """Simple fixed-fractional sizing — easier to reason about for small accounts."""
    fraction = min(fraction, max_fraction)
    capital_for_trade = equity * fraction
    shares = int(capital_for_trade / current_price)
    return max(shares, 0)


def kelly_fraction(win_rate, avg_win, avg_loss):
    """
    Kelly criterion — theoretical optimal bet size.

    f* = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)

    Use half-Kelly in practice (f* × 0.5) to account for estimation error.
    Returns a value clamped to [0, 0.25].
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss          # win/loss ratio
    f = (b * win_rate - (1 - win_rate)) / b
    half_kelly = max(0.0, f * 0.5)
    return min(half_kelly, 0.25)    # never bet more than 25% of equity


def compute_atr_stop(entry_price, atr_value, multiplier=2.0, direction='long'):
    """
    Returns the hard stop-loss price level for a given entry.

    Long  → stop is BELOW entry (price must not fall this far)
    Short → stop is ABOVE entry (price must not rise this far)
    """
    if direction == 'long':
        return entry_price - multiplier * atr_value
    else:   # short
        return entry_price + multiplier * atr_value


def compute_trailing_stop(highest_price, atr_value, multiplier=1.5):
    """
    ATR-based trailing stop for LONG positions.
    Trails the highest price seen since entry; tightens as the trade
    moves in our favour.

        stop = highest_price − multiplier × ATR
    """
    return highest_price - multiplier * atr_value


def compute_short_trailing_stop(lowest_price, atr_value, multiplier=1.5):
    """
    ATR-based trailing stop for SHORT positions.
    Trails the lowest price seen since entry; the stop rises (tightens)
    as the trade moves in our favour (i.e. as price falls).

        stop = lowest_price + multiplier × ATR

    Exit the short when the current price climbs back above this level.
    """
    return lowest_price + multiplier * atr_value


def check_daily_loss_limit(daily_pnl, equity, max_daily_loss_pct=0.02):
    """
    Circuit breaker — returns True if trading should stop for the day.
    Prevents catastrophic drawdowns from compounding intraday.
    """
    return daily_pnl < -(equity * max_daily_loss_pct)