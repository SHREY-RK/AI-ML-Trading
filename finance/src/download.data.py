import pandas as pd
from datetime import datetime, timedelta
import time
from login import generate_session

def download_massive_data(api, symbol, days_back=180, interval=5):
    print(f"Downloading {days_back} days of {interval}m data for {symbol}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    all_data = []
    
    # Backtesting API allows up to 30 days for 5-minute candles.
    chunk_size = timedelta(days=5)
    current_start = start_date
    
    while current_start < end_date:
        current_end = current_start + chunk_size
        if current_end > end_date:
            current_end = end_date
            
        str_start = current_start.strftime("%Y-%m-%d %H:%M:%S")
        str_end = current_end.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Fetching chunk: {str_start} to {str_end}")
        
        try:
            # FIX 1: Updated method name and keyword arguments
            response = api.get_historical_candles(
                groww_symbol=f"NSE-{symbol}",  # Uses NSE-RELIANCE format
                exchange=api.EXCHANGE_NSE,
                segment=api.SEGMENT_CASH,
                start_time=str_start,
                end_time=str_end,
                candle_interval=f"{interval}minute" # Uses "5minute" format
            )
            
            if response and response.get('candles'):
                # FIX 2: Added 'oi' back to the columns list to catch the 7th null item
                df_chunk = pd.DataFrame(
                    response['candles'], 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
                )
                all_data.append(df_chunk)
            
            # Pause for 1 second so Groww doesn't block us for spamming
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching chunk: {e}")
            
        current_start = current_end
        
    if not all_data:
        print("Failed to fetch any data.")
        return
        
    # Combine all chunks into one massive DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Remove any duplicates and sort by time
    final_df.drop_duplicates(subset=['timestamp'], inplace=True)
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
    final_df.sort_values('timestamp', inplace=True)
    
    # Clean up by dropping the empty 'oi' column since we don't need it for CASH
    final_df.drop(columns=['oi'], inplace=True)
    
    # Save to a CSV file!
    filename = f"{symbol}_massive_{interval}m.csv"
    final_df.to_csv(filename, index=False)
    print(f"\n✅ SUCCESS! Saved {len(final_df)} rows to {filename}")

if __name__ == "__main__":
    api = generate_session()
    if api:
        # Downloads 6 months of data
        download_massive_data(api, "RELIANCE", days_back=180, interval=1)