import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import json

def screen_stocks():
    """
    Fetch stocks from Yahoo Finance based on predefined screening criteria.
    """
    try:
        # Define the screening criteria
        q = EquityQuery('and', [
            EquityQuery('gt', ['percentchange', 20]),  # Stocks with >20% change
            EquityQuery('eq', ['region', 'us']),       # US region
            EquityQuery('is-in', ['exchange', 'NMS', 'NYQ', 'NGM', 'ASE', 'PCX', 'YHD', 'NCM']),  # Specific exchanges
        ])
        
        # Fetch the screened stocks
        response = yf.screen(q, sortField='percentchange', sortAsc=True, size=200)


        # Return the list of stocks
        if response and 'quotes' in response:
            return response['quotes']
        else:
            print("No stocks found matching the criteria.")
            return []
    except Exception as e:
        print(f"Error fetching stocks: {e}")
        return []
    
def fetch_historical_data(symbol, period='3d', interval='1d'):
    """
    Fetch historical data for a given symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        return hist
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None
    
def fetch_historical_data(symbol, period='1mo', interval='1d'):
    """
    Fetch historical data for a given symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        print(hist)
        return hist
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None

# Fetch stocks from the predefined screener
screener_stocks = screen_stocks()

# Check if any stocks were found
if screener_stocks:
    # Convert the list of stocks to a DataFrame
    df = pd.DataFrame(screener_stocks)

    # If there is a null value as marketCap, we will drop it
    df = df.dropna(subset=['marketCap'])

    relevant_columns = [
        "symbol",
        "regularMarketPrice",
        "regularMarketChangePercent",
        "marketCap",
        "regularMarketVolume",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "regularMarketDayHigh",
        "regularMarketDayLow",
        "regularMarketPreviousClose",
        "averageDailyVolume3Month"
    ]
    df = df[relevant_columns]

    # Grab the symbols from the screener stocks and print them in a list
    symbols = df['symbol'].tolist()
    for symbol in symbols:
        fetch_historical_data(symbol)

    output_file = 'stock_screener_results.csv'
    df.to_csv(output_file, index=False)
    df.to_json('stock_screener_results.json', indent=4)
    print(f"Combined dataset saved to {output_file}")
    
    # Display the first few rows of the dataset
    print(df.head())
else:
    print("No stocks to save.")