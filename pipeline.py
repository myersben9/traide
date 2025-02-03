import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import json
from typing import List, Dict
import talib as ta
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

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
        data['RSI'] = ta.RSI(close, timeperiod=timeperiod)
        # Get current RSI value
        current_rsi = data['RSI'].iloc[-1]
        return current_rsi
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return 0

def get_stock_list(screener_stocks: List) -> List:
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
    
    # return the symbols in a list
    return df['symbol'].tolist()
def get_moving_averages(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    Calculate short and long moving averages.
    """
    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()
    return data

def get_bollinger_bands(data: pd.DataFrame, window: int, num_std: int) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    """
    try:
        # Ensure 'Close' column exists and is numeric
        if 'Close' not in data.columns:
            raise ValueError("Column 'Close' not found in the DataFrame.")
        
        # Calculate the rolling mean (middle band)
        data['Middle Band'] = data['Close'].rolling(window=window).mean()
        
        # Calculate the rolling standard deviation
        rolling_std = data['Close'].rolling(window=window).std()
        
        # Ensure rolling_std is a Series (not a DataFrame)
        if isinstance(rolling_std, pd.DataFrame):
            rolling_std = rolling_std.squeeze()  # Convert to Series if it's a DataFrame
        
        # Calculate the upper and lower bands
        data['Upper Band'] = data['Middle Band'] + (rolling_std * num_std)
        data['Lower Band'] = data['Middle Band'] - (rolling_std * num_std)
        
        return data
    except Exception as e:
        print(f"Error in get_bollinger_bands: {e}")
        return data

def get_macd(data: pd.DataFrame, fastperiod: int, slowperiod: int, signalperiod: int) -> pd.DataFrame:
    """
    Calculate MACD.
    """
    try:
        # Convert 'Close' column to numpy array and ensure it's 1-dimensional
        close_prices = data['Close'].to_numpy().flatten()
        
        # Calculate MACD
        macd, macd_signal, macd_hist = ta.MACD(close_prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        
        # Add MACD values to the DataFrame
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        data['MACD_Hist'] = macd_hist
        
        return data
    except Exception as e:
        print(f"Error calculating MACD: {e}")
        return data

def get_stochastic_oscillator(data: pd.DataFrame, window: int, smooth_window: int) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.
    """
    try:
        # Convert 'High', 'Low', and 'Close' columns to numpy arrays and flatten
        high_prices = data['High'].to_numpy().flatten()
        low_prices = data['Low'].to_numpy().flatten()
        close_prices = data['Close'].to_numpy().flatten()
        
        # Calculate Stochastic Oscillator
        slowk, slowd = ta.STOCH(high_prices, low_prices, close_prices, fastk_period=window, slowk_period=smooth_window, slowd_period=smooth_window)
        
        # Add Stochastic Oscillator values to the DataFrame
        data['%K'] = slowk
        data['%D'] = slowd
        
        return data
    except Exception as e:
        print(f"Error calculating Stochastic Oscillator: {e}")
        return data

def calculate_support_resistance(data: pd.DataFrame) -> Dict:
    """
    Calculate support and resistance levels using recent price data.
    """
    try:
        # Support: Lowest low in the last 20 days

        low = data['Low'].to_numpy().flatten()

        support = low.min()
        
        # Resistance: Highest high in the last 20 days
        high = data['High'].to_numpy().flatten()
        resistance = high.max()
        
        return {
            "support": support,
            "resistance": resistance
        }
    except Exception as e:
        print(f"Error calculating support/resistance: {e}")
        return {"support": 0, "resistance": 0}

def calculate_fibonacci_levels(data: pd.DataFrame) -> Dict:
    """
    Calculate Fibonacci retracement levels based on recent price swing.
    """
    try:
        # Recent high and low
        recent_high = data['High'].to_numpy().flatten().max()
        recent_low = data['Low'].to_numpy().flatten().min()
        
        # Fibonacci levels
        levels = {
            "0%": recent_high,
            "23.6%": recent_high - (recent_high - recent_low) * 0.236,
            "38.2%": recent_high - (recent_high - recent_low) * 0.382,
            "50%": recent_high - (recent_high - recent_low) * 0.5,
            "61.8%": recent_high - (recent_high - recent_low) * 0.618,
            "100%": recent_low
        }
        
        return levels
    except Exception as e:
        print(f"Error calculating Fibonacci levels: {e}")
        return {}

def get_average_volume(data: pd.DataFrame, window: int) -> float:
    """
    Calculate average volume over a given window.
    """

    return data['Volume'].rolling(window=window).mean().iloc[-1]

def plot_data(data: pd.DataFrame, symbol: str, support_resistance: Dict, fibonacci_levels: Dict, signals: Dict):
    """
    Plot stock data with technical indicators and save to graphs/topmovers/{symbol}.png.
    """
    # Create directories if they don't exist
    output_dir = "graphs/topmovers"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{symbol}.png")

    plt.figure(figsize=(14, 12))

    # Subplot 1: Close Price & Moving Averages
    plt.subplot(4, 1, 1)
    plt.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.5)
    plt.plot(data.index, data['SMA_Short'], label='SMA (10)', color='orange', linestyle='dashed')
    plt.plot(data.index, data['SMA_Long'], label='SMA (50)', color='green', linestyle='dotted')
    
    # Plot support and resistance
    support = support_resistance["support"]
    resistance = support_resistance["resistance"]
    plt.axhline(support, color='red', linestyle='--', label='Support')
    plt.axhline(resistance, color='green', linestyle='--', label='Resistance')
    
    # Plot Buy/Sell signals
    if signals.get('buy_signal', False):
        plt.scatter(data.index[-1], signals['buy_level'], marker='^', color='green', label='Buy Signal', zorder=5)
    if signals.get('sell_signal', False):
        plt.scatter(data.index[-1], signals['sell_level'], marker='v', color='red', label='Sell Signal', zorder=5)

    plt.title(f'{symbol} - Price & Moving Averages', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Subplot 2: Bollinger Bands
    plt.subplot(4, 1, 2)
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['Middle Band'], label='Middle Band', color='red')
    plt.plot(data.index, data['Upper Band'], label='Upper Band', color='green')
    plt.plot(data.index, data['Lower Band'], label='Lower Band', color='purple')

    # Plot Fibonacci levels
    for level, price in fibonacci_levels.items():
        color = 'purple' if level == "100%" else 'cyan'  # Different colors for different levels
        linestyle = '--' if level != "100%" else '-.'  # Dashed lines for non-100% levels
        plt.axhline(price, linestyle=linestyle, color=color, linewidth=2, alpha=0.7, label=f'Fib {level}')
    
    plt.title(f'{symbol} - Bollinger Bands & Fibonacci Levels', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Subplot 3: MACD
    plt.subplot(4, 1, 3)
    plt.plot(data.index, data['MACD'], label='MACD', color='blue', linewidth=1.5)
    plt.plot(data.index, data['MACD_Signal'], label='MACD Signal', color='red', linestyle='dashed')
    plt.bar(data.index, data['MACD_Hist'], label='MACD Hist', color='gray', alpha=0.5)
    plt.title(f'{symbol} - MACD Indicator', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Subplot 4: Stochastic Oscillator
    plt.subplot(4, 1, 4)
    plt.plot(data.index, data['%K'], label='%K', color='magenta', linewidth=1.5)
    plt.plot(data.index, data['%D'], label='%D', color='cyan', linestyle='dashed')
    plt.axhline(80, linestyle='dashed', color='gray', alpha=0.5)  # Overbought level
    plt.axhline(20, linestyle='dashed', color='gray', alpha=0.5)  # Oversold level
    plt.title(f'{symbol} - Stochastic Oscillator', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Stochastic Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_file}")


def generate_signals(signals: Dict, support_resistance: Dict, fibonacci_levels: Dict) -> Dict:
    """
    Generate buy/sell signals based on multiple technical indicators.
    """
    try:
        # Initialize buy and sell signals
        buy_signal = False
        sell_signal = False
        rsi = signals['rsi']
        macd = signals['MACD']
        macd_signal = signals['MACD_Signal']
        close = signals['Close']
        upper_band = signals['Upper Band']
        lower_band = signals['Lower Band']
        support = support_resistance["support"]
        resistance = support_resistance["resistance"]
        adx = signals['adx']



        # 1. RSI Strategy
        if rsi < 30:  # Oversold condition
            buy_signal = True
        elif rsi > 70:  # Overbought condition
            sell_signal = True

        # 2. MACD Crossover Strategy
        if macd > macd_signal:  # Bullish crossover
            buy_signal = True
        elif macd < macd_signal:  # Bearish crossover
            sell_signal = True

        # 3. Bollinger Band Breakout Strategy
        if close > upper_band:  # Price breaks above the upper band
            buy_signal = True
        elif close < lower_band:  # Price breaks below the lower band
            sell_signal = True

        # 4. Support/Resistance Levels Strategy
        if close > support and close < resistance:  # Price is within the support/resistance range
            if close == support:
                buy_signal = True
            if close == resistance:
                sell_signal = True

        # 5. ADX Confirmation
        if adx > 25:  # Strong trend
            if buy_signal and macd > macd_signal:
                buy_signal = True
            if sell_signal and macd < macd_signal:
                sell_signal = True

        return {
            "buy_signal": buy_signal,
            "sell_signal": sell_signal
        }
    except Exception as e:
        print(f"Error generating signals: {e}")
        return {"buy_signal": False, "sell_signal": False}

def main(percentage_change: int):
    # Fetch stocks from the predefined screener
    movers = get_daily_movers(percentage_change)
    if movers:
        symbols = get_stock_list(movers)
    else:
        print("No stocks found matching the criteria.")
        return

    for symbol in symbols:
        # Short Term
        # Download short-term data (1-minute) for the current day
        short_term_data = get_stock_download(symbol, '1d', '1m')  

        # Download slightly longer-term data (5-minute) for the current day
        medium_term_data = get_stock_download(symbol, '1d', '5m')  

        # Download longer-term data (daily) for 2 months
        long_term_data = get_stock_download(symbol, '2mo', '1d')
        if short_term_data is not None and long_term_data is not None:
            # Calculate additional technical indicators for short-term data
            short_term_data = get_moving_averages(short_term_data, 10, 50)
            short_term_data = get_bollinger_bands(short_term_data, 20, 2)
            short_term_data = get_macd(short_term_data, 12, 26, 9)
            short_term_data = get_stochastic_oscillator(short_term_data, 14, 3)
            
            # Calculate support/resistance and Fibonacci levels based on long-term data
            support_resistance = calculate_support_resistance(long_term_data)
            fibonacci_levels = calculate_fibonacci_levels(long_term_data)
        

            # Calculate other existing indicators for different timeframes
            roc = get_roc(short_term_data, 14)  # ROC uses short-term data
            rsi = get_rsi(short_term_data, 14)  # RSI uses short-term data
            adx = get_adx(long_term_data, 14)  # ADX uses medium-term data (5-minute)

            # Add buy/sell levels based on support/resistance and Fibonacci from long-term data
            buy_level = support_resistance["support"]
            sell_level = support_resistance["resistance"]
            
            # Plot the data for visualization
            
            # Append the calculated values for each symbol to the top_movers list

            signals = {
                "symbol": symbol,
                "roc": roc,
                "rsi": rsi,
                "adx": adx,
                "SMA_Short": short_term_data['SMA_Short'].iloc[-1],
                "SMA_Long": short_term_data['SMA_Long'].iloc[-1],
                "MACD": short_term_data['MACD'].iloc[-1],
                "MACD_Signal": short_term_data['MACD_Signal'].iloc[-1],
                "MACD_Hist": short_term_data['MACD_Hist'].iloc[-1],
                "%K": short_term_data['%K'].iloc[-1],
                "%D": short_term_data['%D'].iloc[-1],
                "buy_level": buy_level,
                "sell_level": sell_level,
                "fibonacci_levels": fibonacci_levels,
                "Close": short_term_data['Close'].iloc[-1],
                "Volume": short_term_data['Volume'].iloc[-1],
                "Upper Band": short_term_data['Upper Band'].iloc[-1],
                "Lower Band": short_term_data['Lower Band'].iloc[-1],
                "MACD": short_term_data['MACD'].iloc[-1],


            }

            signals = generate_signals(signals, support_resistance, fibonacci_levels)

            plot_data(short_term_data, symbol, support_resistance, fibonacci_levels, signals)
            

            top_movers.append(signals)

    # Output the top movers to a csv file
    output_file = 'top_movers.csv'
    df = pd.DataFrame(top_movers)
    df.to_csv(output_file, index=False)
    print(f"Top movers saved to {output_file}")
    return top_movers



# Run the monitor
if __name__ == "__main__":
    top_movers = main(80)
