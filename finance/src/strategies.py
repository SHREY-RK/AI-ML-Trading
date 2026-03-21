def generate_signals(df):
    df['signal'] = 0
    df.loc[(df['ema9'] > df['ema21']) & (df['rsi'] < 70), 'signal'] = 1  # Buy
    df.loc[(df['ema9'] < df['ema21']) & (df['rsi'] > 30), 'signal'] = -1  # Sell
    return df
