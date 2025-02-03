import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import json
from typing import List, Dict
import talib as ta
import time
import numpy as np

# List to store top movers
symbols = ['HCWB', 'UDKA', 'GHRS', 'BHAT', 'REBN', 'SOPA', 'BCTX', 'TGI', 'IVVD', 'KC']

def get_daily_movers(percentage_change: int) -> List:
    """
    Fetch stocks from Yahoo Finance based on predefined screening criteria.
    """
    try:
        
        # Define the screening criteria
        q = EquityQuery('and', [
            EquityQuery('gt', ['percentchange', percentage_change]),  # Stocks with >20% change
            EquityQuery('eq', ['region', 'us']),       # US region
            EquityQuery('is-in', ['exchange', 'NMS', 'NYQ', 'NGM', 'ASE', 'PCX', 'YHD', 'NCM']),  # Specific exchanges
        ])
        
        # Fetch the screened stocks
        response = yf.screen(q, sortField='percentchange', sortAsc=True, size=250)

        # Return the list of stocks
        if response and 'quotes' in response:
            return response['quotes']
        else:
            print("No stocks found matching the criteria.")
            return []
    except Exception as e:
        print(f"Error fetching stocks: {e}")
        return []

def download_to_json(data, filename) -> None:
    """
    Save the data to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dumps(data, f, indent=4)
    

# Download historical data for a given symbol
def get_stock_download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch historical data for a given symbol.
    """
    try:
        data = yf.download(symbol, period=period, interval=interval)
        return data
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")

# Function to calculate momentum metrics
def get_roc(data: pd.DataFrame, timeperiod : int) -> pd.DataFrame:
    """
    Calculate momentum metrics for a given stock's historical data.
    """
    # Convert DataFrame columns to NumPy arrays with dtype=float64
    try:
        close = data["Close"].to_numpy().flatten()
        # Calculate ROC
        data['ROC'] = ta.ROC(close, timeperiod=timeperiod)
        # Get current ROC value
        current_roc = data['ROC'].iloc[-1]
        return current_roc
    except Exception as e:
        print(f"Error calculating ROC: {e}")
        return 0
    
def get_adx(data: pd.DataFrame, timeperiod: int) -> pd.DataFrame:
    """
    Calculate ADX for a given stock's historical data.
    """
    try:
        high = data["High"].to_numpy().flatten()
        low = data["Low"].to_numpy().flatten()
        close = data["Close"].to_numpy().flatten()
        data['ADX'] = ta.ADX(high, low, close, timeperiod=timeperiod)

        output_file = 'adx.csv'
        outputdict = {
            'high': high,
            'low': low,
            'close': close,
            'ADX': data['ADX']
        }
        df = pd.DataFrame(outputdict)
        df.to_csv(output_file, index=False)



        # Get current ADX value
        current_adx = data['ADX'].iloc[-1]
        return current_adx
    
    except Exception as e:
        print(f"Error calculating ADX: {e}")
        return 0
    
# Function to calculate RSI 
def get_rsi(data: pd.DataFrame, timeperiod: int) -> pd.DataFrame:
    """
    Calculate RSI for a given stock's historical data.
    """
    try:
        close = data["Close"].to_numpy().flatten()
        print(close)
        data['RSI'] = ta.RSI(close, timeperiod=timeperiod)
        # Get current RSI value
        current_rsi = data['RSI'].iloc[-1]
        return current_rsi
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return 0

def main(symbols: List) -> List:
    # Fetch stocks from the predefined screener

    # List to store top movers
    daily_moves = []

    for symbol in symbols:
        data = get_stock_download(symbol, '2mo', '1d')
        if data is not None:
            roc = get_roc(data, 14)
            rsi = get_rsi(data, 14)
            adx = get_adx(data, 14)
            daily_moves.append({
                "symbol": symbol,
                # "name": stock['shortName'] if 'shortName' in stock else '',
                "roc": roc,
                "rsi": rsi,
                "adx": adx,
            })

    # Output the top movers to a csv file
    output_file = 'top_movers.csv'
    df = pd.DataFrame(daily_moves)
    df.to_csv(output_file, index=False)
    print(f"Top movers saved to {output_file}")


# Run the monitor
if __name__ == "__main__":
    main(symbols)
