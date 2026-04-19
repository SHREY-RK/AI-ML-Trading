import yfinance as yf
import pandas as pd

def fetch_candles(stock_file):
    
    # print(f"Fetching {interval} data for {ticker} from Yahoo Finance...")
    
    # Download data
    # df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    # df = pd.read_csv("META_1min_sample.csv")  # For testing without hitting Yahoo Finance repeatedly
    df = pd.read_csv(f"finance/Data/{stock_file}")  # For testing without hitting Yahoo Finance repeatedly
    
    # if df.empty:
        # raise Exception(f"Failed to fetch data for {ticker}. Check the dates or ticker symbol.")
    
    # # yfinance sometimes returns MultiIndex columns. This flattens them.
    # if isinstance(df.columns, pd.MultiIndex):
    #     df.columns = df.columns.get_level_values(0)
        
    # Reset index to make the date/time a standard column
    # df.reset_index(inplace=True)
    
    # Rename columns to lowercase to perfectly match your existing indicators & backtest code
    # df.rename(columns={
    #     'Date': 'timestamp',
    #     'Open': 'open', 
    #     'High': 'high', 
    #     'Low': 'low', 
    #     'Close': 'close', 
    #     'Volume': 'volume'
    # }, inplace=True)
    print(f"  Successfully fetched {len(df)} rows of data.")
    
    return df