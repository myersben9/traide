


import yfinance
import pandas as pd
import json

# Get specific stock info

def get_stock_info(ticker: str) -> dict:
    """
    Fetch historical data and calculate momentum metrics for a given stock.
    """
    try:
        stock = yfinance.Ticker(ticker)
        return stock.info
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    
# Output into json file

def output_json(data: dict, filename: str) -> None:
    """
    Save the data to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

output_json(get_stock_info('TECX'), 'stock_info.json')