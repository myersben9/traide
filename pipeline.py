import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import json
from typing import List, Dict
import talib as ta
import time
import numpy as np

# List to store top movers
top_movers = []

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
        json.dump(data, f, indent=4)
    

# Function to fetch and process stock data
def get_stock_info(ticker: str) -> Dict:
    """
    Fetch historical data and calculate momentum metrics for a given stock.
    """
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

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
def get_roc(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate momentum metrics for a given stock's historical data.
    """
    # Convert DataFrame columns to NumPy arrays with dtype=float64
    try:
        close = data["Close"].to_numpy().flatten()


        # # Calculate ROC
        data['ROC'] = ta.ROC(close, timeperiod=10)

        # Get current ROC value
        current_roc = data['ROC'].iloc[-1]
    
        return current_roc
    except Exception as e:
        print(f"Error calculating ROC: {e}")
        return 0
    
def get_adx(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ADX for a given stock's historical data.
    """
    try:
        high = data["High"].to_numpy().flatten()
        low = data["Low"].to_numpy().flatten()
        close = data["Close"].to_numpy().flatten()
        data['ADX'] = ta.ADX(high, low, close, timeperiod=14)
        # Get current ADX value
        print(data['ADX'])
        current_adx = data['ADX'].iloc[-1]
        return current_adx
    except Exception as e:
        print(f"Error calculating ADX: {e}")
        return 0
    
# Function to calculate RSI 
def get_rsi(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RSI for a given stock's historical data.
    """
    try:
        close = data["Close"].to_numpy().flatten()
        data['RSI'] = ta.RSI(close, timeperiod=14)
        # Get current RSI value
        current_rsi = data['RSI'].iloc[-1]
        return current_rsi
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return 0

def output_top_stocks(screener_stocks: List) -> None:
    """
    Output the top movers to a file.
    """
    # Convert to dataframe
    df = pd.DataFrame(screener_stocks)
    output_file = 'daily_movers.csv'

    relevant_columns = [
        "symbol",
        "shortName",
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
    df = df.dropna(subset=['marketCap'])
    # Save to CSV
    df.to_csv(output_file, index=False)

def main(percentage_change: int):
    # Fetch stocks from the predefined screener
    movers = get_daily_movers(percentage_change)
    if movers:
        output_top_stocks(movers)
    else:
        print("No stocks found matching the criteria.")
        return

    for stock in movers:
        symbol = stock['symbol']
        data = get_stock_download(symbol, '1mo', '1d')
        if data is not None:
            roc = get_roc(data)
            rsi = get_rsi(data)
            top_movers.append({
                "symbol": symbol,
                "roc": roc,
                "rsi": rsi ,
                "adx": get_adx(data)
            })

    # Output the top movers to a csv file
    output_file = 'top_movers.csv'
    df = pd.DataFrame(top_movers)
    df.to_csv(output_file, index=False)
    print(f"Top movers saved to {output_file}")
    return top_movers


# Run the monitor
if __name__ == "__main__":
    top_movers = main(50)

# def get_search_and_news(symbol : str) :
#     """
#     Fetch historical data for a given symbol.
#     """
#     try:
#         # get list of quotes
#         quotes = yf.Search(symbol, max_results=10).quotes

#         # get list of news
#         news = yf.Search(symbol, news_count=10).news

#         # get list of related research
#         research = yf.Search(symbol, include_research=True).research
#         return quotes, news, research
#     except Exception as e:
#         print(f"Error fetching historical data for {symbol}: {e}")
#         return None

# Fetch stocks from the predefined screener
# screener_stocks = get_daily_movers(5)

# # Check if any stocks were found
# if screener_stocks:
#     # Convert the list of stocks to a DataFrame
#     df = pd.DataFrame(screener_stocks)

#     # If there is a null value as marketCap, we will drop it
#     # df = df.dropna(subset=['marketCap'])

#     relevant_columns = [
#         "symbol",
#         "shortName",
#         "regularMarketPrice",
#         "regularMarketChangePercent",
#         "marketCap",
#         "regularMarketVolume",
#         "fiftyTwoWeekHigh",
#         "fiftyTwoWeekLow",
#         "regularMarketDayHigh",
#         "regularMarketDayLow",
#         "regularMarketPreviousClose",
#         "averageDailyVolume3Month"
#     ]
#     df = df[relevant_columns]

#     # Grab the symbols from the screener stocks and print them in a list
#     symbols = df['symbol'].tolist()

#     for symbol in symbols[:1]:
#         quotes, news, research = get_search_and_news(symbol)
#         if quotes:
#             download_to_json(quotes, f"{symbol}_quotes.json")
#         if news:
#             download_to_json(news, f"{symbol}_news.json")
#         if research:
#             download_to_json(research, f"{symbol}_research.json")

    # output_file = 'stock_screener_results.csv'
    # df.to_csv(output_file, index=False)
    # df.to_json('stock_screener_results.json', indent=4)
    # print(f"Combined dataset saved to {output_file}")
    
    # # Display the first few rows of the dataset
    # print(df.head())
# else:
#     print("No stocks to save.")


# Create algo that checks which stocks are going up the fastest from the daily movers