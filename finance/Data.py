from SmartApi import SmartConnect
import pyotp
import pandas as pd
import requests
from config import API_KEY, CLIENT_CODE, MPIN, TOTP_SECRET

try:
    if not all([API_KEY, CLIENT_CODE, MPIN, TOTP_SECRET]):
        raise Exception("Missing credentials in .env file")
    
    smartApi = SmartConnect(API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    
    session = smartApi.generateSession(CLIENT_CODE, MPIN, totp)
    
    if not session or session.get('status') == False:
        raise Exception(f"Login failed: {session.get('message', 'Unknown error')}")
    
    print("Session created successfully")
    print(f"Auth Token: {session['data']['jwtToken'][:20]}...")
    
    params = {
        "exchange": "NSE",
        "symboltoken": "11536",
        "interval": "ONE_MINUTE",
        "fromdate": "2026-02-20 09:15",
        "todate": "2026-02-20 15:30"
    }
    
    # Get stock name from Angel One master data with caching
    _token_cache = {}
    
    def get_stock_name(token, exchange="NSE"):
        if not _token_cache:
            url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            try:
                data = requests.get(url).json()
                for item in data:
                    key = f"{item['exch_seg']}_{item['token']}"
                    _token_cache[key] = item['symbol']
            except:
                pass
        
        key = f"{exchange}_{token}"
        return _token_cache.get(key, f"Token-{token}")
    
    stock_name = get_stock_name(params['symboltoken'], params['exchange'])
    
    response = smartApi.getCandleData(params)
    
    if response['status']:
        df = pd.DataFrame(response['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"\nCandle Data for {stock_name} (Token: {params['symboltoken']}):")
        print(df)
    else:
        print(f"Failed to fetch data: {response['message']}")
    
except Exception as e:
    print(f"Error: {e}")