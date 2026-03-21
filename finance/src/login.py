from SmartApi import SmartConnect
import pyotp
from config import API_KEY, CLIENT_CODE, MPIN

def generate_session():
    try:
        smartApi = SmartConnect(API_KEY)
        totp = pyotp.TOTP("S6QN3EBYUFCLLYAYB6PY3DWI4E").now()
        data = smartApi.generateSession(CLIENT_CODE, MPIN, totp)
        
        if not data or not data.get("status"):
            raise Exception(f"Login failed: {data.get('message', 'Unknown error')}")
        
        print("Login Successful")
        return smartApi
        
    except Exception as e:
        print("Session generation failed:", e)
        return None

if __name__ == "__main__":
    api = generate_session()