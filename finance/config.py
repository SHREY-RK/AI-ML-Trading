import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
MPIN = os.getenv("MPIN")
TOTP_SECRET = os.getenv("TOTP_SECRET")

if not all([API_KEY, CLIENT_CODE, MPIN, TOTP_SECRET]):
    raise ValueError("Missing environment variables. Check .env file.")