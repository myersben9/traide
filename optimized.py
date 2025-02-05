import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Dict, Optional
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    'percentage_change': 20,  # Lower threshold for day trading
    'short_window': 10,      # Short-term moving average
    'long_window': 50,       # Long-term moving average
    'bollinger_window': 20,  # Bollinger Bands window
    'bollinger_std': 2,      # Bollinger Bands standard deviation
    'macd_fast': 12,         # MACD fast period
    'macd_slow': 26,         # MACD slow period
    'macd_signal': 9,        # MACD signal period
    'stochastic_window': 14, # Stochastic Oscillator window
    'stochastic_smooth': 3,  # Stochastic smoothing
    'rsi_period': 14,        # RSI period
    'adx_period': 14,        # ADX period
    'roc_period': 14,        # Rate of Change period
    'volume_window': 20,     # Volume moving average window
    'vwap_window': 20,       # VWAP window
    'output_dir': os.path.join('graphs', 'topmovers'),  # Output directory for graphs
    'daily_movers_file': 'daily_movers.csv',            # File to save daily movers
    'top_movers_file': 'top_movers.csv',                # File to save top movers
    'periods': {
        'intraday': {'period': '1d', 'interval': '5m'},  # Intraday data (5-minute intervals)
        'short_term': {'period': '5d', 'interval': '1h'} # Short-term data (hourly intervals)
    }
}


def get_daily_movers(percentage_change: int) -> List[Dict]:
    """
    Fetch stocks from Yahoo Finance based on predefined screening criteria.
    """
    try:
        q = yf.EquityQuery('and', [
            yf.EquityQuery('gt', ['percentchange', percentage_change]),
            yf.EquityQuery('eq', ['region', 'us']),
            yf.EquityQuery('is-in', ['exchange', 'NMS', 'NYQ', 'NGM', 'ASE', 'PCX', 'YHD', 'NCM']),
        ])
        response = yf.screen(q, sortField='percentchange', sortAsc=True, size=250)
        return response['quotes'] if response and 'quotes' in response else []
    except Exception as e:
        logging.error(f"Error fetching stocks: {e}")
        return []

def save_to_csv(data: List[Dict], filename: str) -> None:
    """
    Save data to a CSV file.
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"Data saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving data to {filename}: {e}")

def download_stock_data(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for a given symbol.
    """
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        return data if not data.empty else None
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return None

def calculate_technical_indicators(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculate various technical indicators for the given stock data.
    """
    try:
        # Ensure all columns are 1-dimensional
        close = data['Close'].to_numpy().flatten()
        high = data['High'].to_numpy().flatten()
        low = data['Low'].to_numpy().flatten()
        volume = data['Volume'].to_numpy().flatten()

        # Check if there's enough data to calculate indicators
        if len(close) < CONFIG['long_window']:
            logging.warning(f"Insufficient data to calculate indicators for this stock.")
            return None

        # Calculate indicators
        data['SMA_Short'] = data['Close'].rolling(window=CONFIG['short_window']).mean()
        data['SMA_Long'] = data['Close'].rolling(window=CONFIG['long_window']).mean()

        # Bollinger Bands
        middle_band = data['Close'].rolling(window=CONFIG['bollinger_window']).mean()
        rolling_std = data['Close'].rolling(window=CONFIG['bollinger_window']).std()
        data['Middle Band'] = middle_band
        data['Upper Band'] = (middle_band + (rolling_std * CONFIG['bollinger_std']))
        data['Lower Band'] = (middle_band - (rolling_std * CONFIG['bollinger_std']))

        # MACD
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=CONFIG['macd_fast'], slowperiod=CONFIG['macd_slow'], signalperiod=CONFIG['macd_signal'])
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        data['MACD_Hist'] = macd_hist

        # Stochastic Oscillator
        slowk, slowd = ta.STOCH(high, low, close, fastk_period=CONFIG['stochastic_window'], slowk_period=CONFIG['stochastic_smooth'], slowd_period=CONFIG['stochastic_smooth'])
        data['%K'] = slowk
        data['%D'] = slowd

        # RSI
        data['RSI'] = ta.RSI(close, timeperiod=CONFIG['rsi_period'])

        # ADX
        data['ADX'] = ta.ADX(high, low, close, timeperiod=CONFIG['adx_period'])

        # ROC
        data['ROC'] = ta.ROC(close, timeperiod=CONFIG['roc_period'])

        # Average Volume
        data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

        return data
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return None

def calculate_support_resistance(data: pd.DataFrame) -> Dict:
    """
    Calculate support and resistance levels.
    """
    try:

        low = data['Low'].to_numpy().flatten()
        high = data['High'].to_numpy().flatten()

        return {
            "support": low.min(),
            "resistance": high.max()
        }
    except Exception as e:
        logging.error(f"Error calculating support/resistance: {e}")
        return {"support": 0, "resistance": 0}

def calculate_fibonacci_levels(data: pd.DataFrame) -> Dict:
    """
    Calculate Fibonacci retracement levels.
    """
    try:
        recent_high = data['High'].max()
        recent_low = data['Low'].min()
        return {
            "0%": recent_high,
            "23.6%": recent_high - (recent_high - recent_low) * 0.236,
            "38.2%": recent_high - (recent_high - recent_low) * 0.382,
            "50%": recent_high - (recent_high - recent_low) * 0.5,
            "61.8%": recent_high - (recent_high - recent_low) * 0.618,
            "100%": recent_low
        }
    except Exception as e:
        logging.error(f"Error calculating Fibonacci levels: {e}")
        return {}

def generate_signals(data: pd.DataFrame, support_resistance: Dict) -> Dict:
    """
    Generate buy/sell signals based on technical indicators.
    """
    try:
        buy_signal = False
        sell_signal = False

        #output the data to a csv file

        close = data['Close'].to_numpy().flatten()
        high = data['High'].to_numpy().flatten()
        rsi = data['RSI'].to_numpy().flatten()
        macd = data['MACD'].to_numpy().flatten()
        macd_signal = data['MACD_Signal'].to_numpy().flatten()
        upper_band = data['Upper Band'].to_numpy().flatten()
        lower_band = data['Lower Band'].to_numpy().flatten()
        support = support_resistance["support"]
        resistance = support_resistance["resistance"]
        adx = data['ADX'].to_numpy().flatten()
        vnap = data['VWAP'].to_numpy().flatten()



        # Use the latest values for calculations
        close = close[-1]
        rsi = rsi[-1]
        macd = macd[-1]
        macd_signal = macd_signal[-1]
        upper_band = upper_band[-1]
        lower_band = lower_band[-1]
        adx = adx[-1]
        

        support = support_resistance["support"]
        resistance = support_resistance["resistance"]
        


        print("close", close)
        print("rsi", rsi)
        print("macd", macd)
        print("macd_signal", macd_signal)
        print("upper_band", upper_band)
        print("lower_band", lower_band)
        print("support", support)
        print("resistance", resistance)
        print("adx", adx)



        # Generate signals
        if rsi < 30:
            buy_signal = True
        elif rsi > 70:
            sell_signal = True

        if macd > macd_signal:
            buy_signal = True
        elif macd < macd_signal:
            sell_signal = True

        if close > upper_band:
            buy_signal = True
        elif close < lower_band:
            sell_signal = True

        if support <= close <= resistance:
            if close == support:
                buy_signal = True
            if close == resistance:
                sell_signal = True

        if adx > 25:
            if buy_signal and macd > macd_signal:
                buy_signal = True
            if sell_signal and macd < macd_signal:
                sell_signal = True

        return {
            "buy_signal": buy_signal,
            "sell_signal": sell_signal
        }
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        return {"buy_signal": False, "sell_signal": False}

def plot_data(data: pd.DataFrame, symbol: str, support_resistance: Dict, fibonacci_levels: Dict, signals: Dict) -> None:
    """
    Plot stock data with technical indicators and save to file.
    """
    try:
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        output_file = os.path.join(CONFIG['output_dir'], f"{symbol}.png")

        plt.figure(figsize=(14, 12))

        # Convert all columns to flattened numpy arrays
        index = data.index.to_numpy()  # Convert index to numpy array
        close = data['Close'].to_numpy().flatten()
        sma_short = data['SMA_Short'].to_numpy().flatten()
        sma_long = data['SMA_Long'].to_numpy().flatten()
        middle_band = data['Middle Band'].to_numpy().flatten()
        upper_band = data['Upper Band'].to_numpy().flatten()
        lower_band = data['Lower Band'].to_numpy().flatten()
        macd = data['MACD'].to_numpy().flatten()
        macd_signal = data['MACD_Signal'].to_numpy().flatten()
        macd_hist = data['MACD_Hist'].to_numpy().flatten()
        percent_k = data['%K'].to_numpy().flatten()
        percent_d = data['%D'].to_numpy().flatten()

        # Subplot 1: Price & Moving Averages
        plt.subplot(4, 1, 1)
        plt.plot(index, close, label='Close Price', color='blue', linewidth=1.5)
        plt.plot(index, sma_short, label=f'SMA ({CONFIG["short_window"]})', color='orange', linestyle='dashed')
        plt.plot(index, sma_long, label=f'SMA ({CONFIG["long_window"]})', color='green', linestyle='dotted')
        plt.axhline(support_resistance["support"], color='red', linestyle='--', label='Support')
        plt.axhline(support_resistance["resistance"], color='green', linestyle='--', label='Resistance')
        print(signals['buy_signal'])
        print(signals['sell_signal'])
        # Print the type and values of the signals to verify they are booleans
        print(f"Type of buy_signal: {type(signals['buy_signal'])}, Value: {signals['buy_signal']}")
        print(f"Type of sell_signal: {type(signals['sell_signal'])}, Value: {signals['sell_signal']}")

        if signals['buy_signal']:
            plt.scatter(index[-1], close[-1], marker='^', color='green', label='Buy Signal', zorder=5)
        if signals['sell_signal']:
            plt.scatter(index[-1], close[-1], marker='v', color='red', label='Sell Signal', zorder=5)
        plt.title(f'{symbol} - Price & Moving Averages', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Subplot 2: Bollinger Bands & Fibonacci Levels
        plt.subplot(4, 1, 2)
        plt.plot(index, close, label='Close Price', color='blue')
        plt.plot(index, middle_band, label='Middle Band', color='red')
        plt.plot(index, upper_band, label='Upper Band', color='green')
        plt.plot(index, lower_band, label='Lower Band', color='purple')
        for level, price in fibonacci_levels.items():
    # Ensure price is a scalar (not a Series)
            price = price.item() if isinstance(price, pd.Series) else price
            color = 'purple' if level == "100%" else 'cyan'
            linestyle = '--' if level != "100%" else '-.'
            plt.axhline(price, linestyle=linestyle, color=color, linewidth=2, alpha=0.7, label=f'Fib {level}')

        plt.title(f'{symbol} - Bollinger Bands & Fibonacci Levels', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Subplot 3: MACD
        plt.subplot(4, 1, 3)
        plt.plot(index, macd, label='MACD', color='blue', linewidth=1.5)
        plt.plot(index, macd_signal, label='MACD Signal', color='red', linestyle='dashed')
        plt.bar(index, macd_hist, label='MACD Hist', color='gray', alpha=0.5)
        plt.title(f'{symbol} - MACD Indicator', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('MACD Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Subplot 4: Stochastic Oscillator
        plt.subplot(4, 1, 4)
        plt.plot(index, percent_k, label='%K', color='magenta', linewidth=1.5)
        plt.plot(index, percent_d, label='%D', color='cyan', linestyle='dashed')
        plt.axhline(80, linestyle='dashed', color='gray', alpha=0.5)
        plt.axhline(20, linestyle='dashed', color='gray', alpha=0.5)
        plt.title(f'{symbol} - Stochastic Oscillator', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Stochastic Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot saved: {output_file}")
    except Exception as e:
        logging.error(f"Error plotting data for {symbol}: {e}")
        traceback.print_exc()  
def main():
    movers = get_daily_movers(CONFIG['percentage_change'])
    if not movers:
        logging.warning("No stocks found matching the criteria.")
        return

    symbols = [mover['symbol'] for mover in movers]
    save_to_csv(movers, CONFIG['daily_movers_file'])

    top_movers = []
    for symbol in symbols:
        logging.info(f"Processing {symbol}...")

        # Download data
        short_term_data = download_stock_data(symbol, '1d', '1m')
        long_term_data = download_stock_data(symbol, '2mo', '1d')

        # Skip if data is missing
        if short_term_data is None or long_term_data is None:
            logging.warning(f"Skipping {symbol}: Missing data.")
            continue

        # Calculate technical indicators
        short_term_data = calculate_technical_indicators(short_term_data)
        if short_term_data is None:
            logging.warning(f"Skipping {symbol}: Insufficient data for technical indicators.")
            continue

        # Calculate support/resistance and Fibonacci levels
        support_resistance = calculate_support_resistance(long_term_data)
        fibonacci_levels = calculate_fibonacci_levels(long_term_data)

        # Generate signals
        signals = generate_signals(short_term_data, support_resistance)

        print(signals)

        # Plot data
        plot_data(short_term_data, symbol, support_resistance, fibonacci_levels, signals)

        # Add to top movers list
        top_movers.append({
            "symbol": symbol,
            "roc": short_term_data['ROC'].iloc[-1],
            "rsi": short_term_data['RSI'].iloc[-1],
            "adx": short_term_data['ADX'].iloc[-1],
            "SMA_Short": short_term_data['SMA_Short'].iloc[-1],
            "SMA_Long": short_term_data['SMA_Long'].iloc[-1],
            "MACD": short_term_data['MACD'].iloc[-1],
            "MACD_Signal": short_term_data['MACD_Signal'].iloc[-1],
            "MACD_Hist": short_term_data['MACD_Hist'].iloc[-1],
            "%K": short_term_data['%K'].iloc[-1],
            "%D": short_term_data['%D'].iloc[-1],
            "buy_level": support_resistance["support"],
            "sell_level": support_resistance["resistance"],
            "fibonacci_levels": fibonacci_levels,
            "Close": short_term_data['Close'].iloc[-1],
            "Volume": short_term_data['Volume'].iloc[-1],
            "Upper Band": short_term_data['Upper Band'].iloc[-1],
            "Lower Band": short_term_data['Lower Band'].iloc[-1],
            **signals
        })

    save_to_csv(top_movers, CONFIG['top_movers_file'])
    logging.info("Top movers analysis completed.")

if __name__ == "__main__":
    main() 