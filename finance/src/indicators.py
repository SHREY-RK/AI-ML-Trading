def ema(df, period):
    return df['close'].ewm(span=period, adjust=False).mean()

def rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def volatility(df, period=20):
    return df['close'].rolling(window=period).std()
