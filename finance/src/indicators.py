import pandas as pd
import numpy as np


def ema(df, period):
    return df['close'].ewm(span=period, adjust=False).mean()


def rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(df, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram)"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df, period=14):
    """Average True Range — measures volatility in price units."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def bollinger_bands(df, period=20, std_dev=2):
    """Returns (upper_band, middle_band, lower_band)"""
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def volatility(df, period=20):
    return df['close'].rolling(window=period).std()


def add_all_indicators(df):
    """Convenience function: adds all indicators to a DataFrame in-place."""
    df['ema9'] = ema(df, 9)
    df['ema21'] = ema(df, 21)
    df['ema50'] = ema(df, 50)
    df['rsi'] = rsi(df, 14)
    df['volatility'] = volatility(df, 20)
    df['atr'] = atr(df, 14)

    macd_line, signal_line, histogram = macd(df)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = histogram

    bb_upper, bb_mid, bb_lower = bollinger_bands(df)
    df['bb_upper'] = bb_upper
    df['bb_mid'] = bb_mid
    df['bb_lower'] = bb_lower

    # Derived / normalised features for ML
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    return df