import yfinance as yf
import pandas as pd
import json
import talib as ta
import numpy as np

# List to store top movers
top_movers = []

def get_daily_movers(percentage_change: int):
    """
    Fetch stocks from Yahoo Finance based on predefined screening criteria.
    """
    try:
        # Fetch stocks based on screening criteria
        q = yf.EquityQuery('and', [
            yf.EquityQuery('gt', ['percentchange', percentage_change]),
            yf.EquityQuery('eq', ['region', 'us']),
            yf.EquityQuery('is-in', ['exchange', 'NMS', 'NYQ', 'NGM', 'ASE', 'PCX', 'YHD', 'NCM']),
        ])
        
        response = yf.screen(q, sortField='percentchange', sortAsc=True, size=250)

        return response['quotes'] if response and 'quotes' in response else []
    except Exception as e:
        print(f"Error fetching stocks: {e}")
        return []

def get_stock_download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch historical data for a given stock.
    """
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.isnull().values.any():
            print(f"Warning: Missing data detected in {symbol}. This may cause NaN values in calculations.")
        return data
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None

def detect_nan_issues(data: pd.DataFrame, indicator_name: str, symbol: str):
    """
    Detects reasons for NaN values in calculations.
    """
    if data.isnull().values.any():
        missing_columns = data.columns[data.isnull().any()].tolist()
        print(f"⚠️ Warning: NaN detected in {indicator_name} for {symbol}. Possible reasons:")
        print(f"   - Missing data in: {missing_columns}")
        print(f"   - Insufficient historical data for calculation")
        print(f"   - Non-numeric values present")
        print(f"   - Recent IPO with limited price history")

def calculate_roc(data: pd.DataFrame, timeperiod: int, symbol: str):
    """
    Calculate Rate of Change (ROC).
    """
    try:
        print(data)
        close = data["Close"].to_numpy().flatten()
        print(close)
        data['ROC'] = ta.ROC(close, timeperiod=timeperiod)
        detect_nan_issues(data, "ROC", symbol)
        return data['ROC'].iloc[-1] if not np.isnan(data['ROC'].iloc[-1]) else 0
    except Exception as e:
        print(f"Error calculating ROC for {symbol}: {e}")
        return 0

def calculate_rsi(data: pd.DataFrame, timeperiod: int, symbol: str):
    """
    Calculate Relative Strength Index (RSI).
    """
    try:
        close = data["Close"].to_numpy().flatten()
        data['RSI'] = ta.RSI(close, timeperiod=timeperiod)
        detect_nan_issues(data, "RSI", symbol)
        return data['RSI'].iloc[-1] if not np.isnan(data['RSI'].iloc[-1]) else 0
    except Exception as e:
        print(f"Error calculating RSI for {symbol}: {e}")
        return 0

def calculate_adx(data: pd.DataFrame, timeperiod: int, symbol: str):
    """
    Calculate Average Directional Index (ADX).
    """
    try:
        high, low, close = data["High"].to_numpy(), data["Low"].to_numpy(), data["Close"].to_numpy()
        data['ADX'] = ta.ADX(high, low, close, timeperiod=timeperiod)
        detect_nan_issues(data, "ADX", symbol)
        return data['ADX'].iloc[-1] if not np.isnan(data['ADX'].iloc[-1]) else 0
    except Exception as e:
        print(f"Error calculating ADX for {symbol}: {e}")
        return 0

def detect_fastest_risers(top_movers):
    """
    Identify and rank the fastest-moving stocks based on ROC and RSI.
    """
    df = pd.DataFrame(top_movers)
    df['Momentum Score'] = df['roc'] + df['rsi']  # Combine momentum indicators
    df = df.sort_values(by='Momentum Score', ascending=False)
    
    print("\n🚀 Fastest Rising Stocks 🚀")
    print(df[['symbol', 'name', 'Momentum Score']].head(10))
    
    return df

def main(percentage_change: int):
    movers = get_daily_movers(percentage_change)
    if not movers:
        print("No stocks found matching the criteria.")
        return

    for stock in movers:
        symbol = stock['symbol']
        data = get_stock_download(symbol, '1mo', '1d')

        if data is not None:
            roc = calculate_roc(data, 14, symbol)
            rsi = calculate_rsi(data, 14, symbol)
            adx = calculate_adx(data, 14, symbol)

            top_movers.append({
                "symbol": symbol,
                "name": stock.get('shortName', ''),
                "roc": roc,
                "rsi": rsi,
                "adx": adx
            })

    # Rank fastest movers
    ranked_df = detect_fastest_risers(top_movers)

    # Save results
    ranked_df.to_csv('top_movers.csv', index=False)
    print("✅ Top movers saved to top_movers.csv")
    return ranked_df

# Run script
if __name__ == "__main__":
    main(50)
