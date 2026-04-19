import os
from growwapi import GrowwAPI
from dotenv import load_dotenv

load_dotenv()

def generate_session():
    try:
        api_key = os.getenv("GROWW_API_KEY")
        secret = os.getenv("GROWW_API_SECRET")

        if not api_key or not secret:
            raise Exception("Missing Groww API Key or Secret in .env file")

        # Generate the daily access token
        access_token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
        
        # Initialize the API client
        groww = GrowwAPI(access_token)
        
        print("YUP! Groww Login Successful")
        return groww
        
    except Exception as e:
        print("Groww Session generation failed:", e)
        return None

if __name__ == "__main__":
    api = generate_session()