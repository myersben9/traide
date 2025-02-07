from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd

# Alpaca API Credentials (Replace with your own keys)
KEY_ID = "PK31EN7WFF54A1GDEF4L"
SECRET_KEY = "xq92EVg4vQny19fQVVOhvzvurDeTwYm4ay4iQrfY"
BASE_URL = "https://paper-api.alpaca.markets"

# Instantiate REST API Connection
api = REST(key_id=KEY_ID,secret_key=SECRET_KEY,base_url="https://paper-api.alpaca.markets")

# Fetch 1Minute historical bars of Bitcoin
bars = api.get_crypto_bars("BTC", TimeFrame.Minute).df
print(bars)