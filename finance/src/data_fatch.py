import pandas as pd

def fetch_candles(api, token, fromdate, todate, exchange="NSE", interval="ONE_MINUTE"):
    params = {
        "exchange": exchange,
        "symboltoken": token,
        "interval": interval,
        "fromdate": fromdate,
        "todate": todate
    }
    
    response = api.getCandleData(params)
    
    if response['status']:
        df = pd.DataFrame(response['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        raise Exception(f"Failed to fetch data: {response['message']}")
