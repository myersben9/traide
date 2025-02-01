import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
import time
from yfinance import EquityQuery
from typing import List

# Define time interval (1-minute for real-time tracking)
INTERVAL = "1m"
PERIOD = "1d"
REFRESH_RATE = 60  # Refresh every 60 seconds

# Fetch top gainers

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

# Fetch historical data

def get_stock_data(symbol):
    try:
        data = yf.download(symbol, period=PERIOD, interval=INTERVAL)
        return data if not data.empty else None
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Calculate momentum indicators

def calculate_momentum_indicators(data: pd.DataFrame) -> pd.Series:
    close = data['Close'].to_numpy().flatten()
    high = data['High'].to_numpy().flatten()
    low = data['Low'].to_numpy().flatten()
    
    data['ROC'] = ta.ROC(close, timeperiod=5)  # Rate of Change
    #fill nan values with 0

    data['MOM'] = ta.MOM(close, timeperiod=5)  # Momentum
    macd, signal, _ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD_Hist'] = macd - signal  # MACD Histogram
    data['ATR'] = ta.ATR(high, low, close, timeperiod=14)  # Volatility
    
    return data.iloc[-1][['ROC', 'MOM', 'MACD_Hist', 'ATR']]

# Main Screener

def real_time_screener():
    while True:
        top_movers = get_daily_movers(50)
        stock_momentum = []
        
        for stock in top_movers:
            data = get_stock_data(stock['symbol'])
            if data is not None:
                indicators = calculate_momentum_indicators(data)
                stock_momentum.append({"Symbol": stock['symbol'], **indicators})
        
        df = pd.DataFrame(stock_momentum)
        df = df.sort_values(by=["ROC", "MOM", "MACD_Hist"], ascending=False)
        print(df.head(10))  # Display top 10 fastest-moving stocks
        
        time.sleep(REFRESH_RATE)

# Run the screener
if __name__ == "__main__":
    real_time_screener()
