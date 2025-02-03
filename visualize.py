import os
import yfinance as yf
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

# Create main graphs directory
base_graph_dir = "graphs"
os.makedirs(base_graph_dir, exist_ok=True)

# Create subdirectories for different chart types
chart_types = ["price", "rsi", "momentum"]
for chart in chart_types:
    os.makedirs(os.path.join(base_graph_dir, chart), exist_ok=True)

# Load the top movers from CSV
top_movers_df = pd.read_csv('top_movers.csv')

def create_explanation_files():
    explanations = {
        "price": "This chart shows the stock price movement over time. A general uptrend suggests buying opportunities, while a downtrend suggests selling or avoiding the stock.",
        "rsi": "The RSI (Relative Strength Index) chart measures momentum. Values above 70 indicate overbought conditions (potential sell), while values below 30 indicate oversold conditions (potential buy).",
        "momentum": "The ROC (Rate of Change) and ADX (Average Directional Index) indicators show trend strength. A rising ADX suggests a strong trend, while ROC shows momentum shifts."
    }
    
    for chart, text in explanations.items():
        with open(os.path.join(base_graph_dir, chart, "explanation.txt"), "w") as f:
            f.write(text)

def visualize_stock(symbol):
    """Fetch stock data and generate price, RSI, ROC, and ADX charts with buy/sell signals."""
    try:
        # Download historical data
        data = yf.download(symbol, period='2mo', interval='1d')

        if data.empty:
            print(f"No data available for {symbol}")
            return
        
        # Calculate indicators
        data['RSI'] = ta.RSI(data['Close'].to_numpy().flatten(), timeperiod=14)
        data['ROC'] = ta.ROC(data['Close'].to_numpy().flatten(), timeperiod=14)
        data['ADX'] = ta.ADX(data['High'].to_numpy().flatten(), data['Low'].to_numpy().flatten(), data['Close'].to_numpy().flatten(), timeperiod=14)
        
        # Identify buy/sell signals
        buy_signals = data[data['RSI'] < 30].index
        sell_signals = data[data['RSI'] > 70].index

        base_graph_dir_price = "graphs" + "/price"
        
        # Save Stock Price Chart
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Stock Price', color='blue')
        plt.scatter(buy_signals, data.loc[buy_signals, 'Close'], marker='^', color='green', label='Buy Signal', s=100)
        plt.scatter(sell_signals, data.loc[sell_signals, 'Close'], marker='v', color='red', label='Sell Signal', s=100)
        plt.title(f"{symbol} Price Movement")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(os.path.join(base_graph_dir, "price", f"{symbol}_price.png"))
        plt.close()

        base_graph_dir_rsi = "graphs" + "/rsi"
        
        # Save RSI Chart
        plt.figure(figsize=(12, 4))
        plt.plot(data.index, data['RSI'], label='RSI', color='green')
        plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
        plt.axhline(30, linestyle='--', color='blue', label='Oversold (30)')
        plt.scatter(buy_signals, data.loc[buy_signals, 'RSI'], marker='^', color='green', label='Buy Signal', s=100)
        plt.scatter(sell_signals, data.loc[sell_signals, 'RSI'], marker='v', color='red', label='Sell Signal', s=100)
        plt.title(f"{symbol} RSI Chart")
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.savefig(os.path.join(base_graph_dir, "rsi", f"{symbol}_rsi.png"))
        plt.close()

        base_graph_dir_momentum = "graphs" + "/momentum"

        # Save ROC and ADX Chart
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['ROC'], label='ROC (Rate of Change)', color='orange')
        plt.plot(data.index, data['ADX'], label='ADX (Trend Strength)', color='purple')
        plt.title(f"{symbol} Momentum Indicators (ROC & ADX)")
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(base_graph_dir, "momentum", f"{symbol}_momentum.png"))
        plt.close()

        print(f"Charts saved for {symbol} in '{base_graph_dir}' folder.")

    except Exception as e:
        print(f"Error generating charts for {symbol}: {e}")

# Generate explanation files
create_explanation_files()

# Generate charts for each top mover
for symbol in top_movers_df['symbol'].tolist():
    visualize_stock(symbol)
