import pandas as pd
import numpy as np
import yfinance as yf
import talib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Settings: Change these as desired
# -------------------------------
stock_symbol = "LTRY"           # Stock to test (e.g., AAPL)
investment_amount = 10000       # Amount in dollars you plan to invest
interval = "5m"                 # Interval for intraday data
period = "1d"                   # Period to fetch (e.g., one day of intraday data)

# -------------------------------
# Directory and Model Path
# -------------------------------
report_dir = "data_report"
model_path = os.path.join(report_dir, "profit_predictor_model.pkl")

# -------------------------------
# Load the Trained Model
# -------------------------------
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}")
model = joblib.load(model_path)
print(f"Loaded trained model from {model_path}")

# -------------------------------
# Define Functions for Data Fetching and Indicator Calculation
# -------------------------------
def fetch_intraday_data(symbol, interval="5m", period="1d"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(interval=interval, period=period)
        # Ensure we only get necessary columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data['Symbol'] = symbol
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df):
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    df["MACD"], df["MACD Signal"], _ = talib.MACD(df["Close"])
    df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    return df

def calculate_additional_indicators(df):
    upperband, middleband, lowerband = talib.BBANDS(df["Close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df["UpperBand"] = upperband
    df["MiddleBand"] = middleband
    df["LowerBand"] = lowerband
    slowk, slowd = talib.STOCH(df["High"], df["Low"], df["Close"],
                               fastk_period=14, slowk_period=3, slowk_matype=0,
                               slowd_period=3, slowd_matype=0)
    df["Stoch_K"] = slowk
    df["Stoch_D"] = slowd
    return df

# -------------------------------
# Updated Function: Identify best trade point with fallback option
# -------------------------------
def find_best_trade_point(df):
    """
    For the given day's intraday data, identify the best trade point:
      - Ideally, buy at the minimum price (provided it occurs before the maximum price)
      - Sell at the maximum price.
    If no valid trade point is found, use the first observation as the buy point
    and the last observation as the sell point.
    Returns a dictionary of trade features.
    """
    df["Date"] = df.index.date
    # Sort by time to ensure correct ordering
    df = df.sort_index()
    groups = df.groupby("Date")
    for date, group in groups:
        if group.empty or len(group) < 2:
            continue
        min_price = group["Low"].min()
        max_price = group["High"].max()
        min_time = group[group["Low"] == min_price].index[0]
        max_time = group[group["High"] == max_price].index[0]
        if min_time < max_time:
            trade_features = {
                "Buy Price": min_price,
                "Sell Price": max_price,
                "Profit": max_price - min_price,
                "RSI_Buy": group.loc[min_time, "RSI"],
                "RSI_Sell": group.loc[max_time, "RSI"],
                "MACD_Buy": group.loc[min_time, "MACD"],
                "MACD_Sell": group.loc[max_time, "MACD"],
                "ATR_Buy": group.loc[min_time, "ATR"],
                "ATR_Sell": group.loc[max_time, "ATR"],
                "Volume_Buy": group.loc[min_time, "Volume"],
                "Volume_Sell": group.loc[max_time, "Volume"],
                "Stoch_K_Buy": group.loc[min_time, "Stoch_K"] if "Stoch_K" in group.columns else np.nan,
                "Stoch_K_Sell": group.loc[max_time, "Stoch_K"] if "Stoch_K" in group.columns else np.nan,
                "Stoch_D_Buy": group.loc[min_time, "Stoch_D"] if "Stoch_D" in group.columns else np.nan,
                "Stoch_D_Sell": group.loc[max_time, "Stoch_D"] if "Stoch_D" in group.columns else np.nan,
            }
            return trade_features
    # Fallback: Use first and last observation if no valid trade point is found
    if not df.empty and len(df) >= 2:
        trade_features = {
            "Buy Price": df["Low"].iloc[0],
            "Sell Price": df["High"].iloc[-1],
            "Profit": df["High"].iloc[-1] - df["Low"].iloc[0],
            "RSI_Buy": df["RSI"].iloc[0],
            "RSI_Sell": df["RSI"].iloc[-1],
            "MACD_Buy": df["MACD"].iloc[0],
            "MACD_Sell": df["MACD"].iloc[-1],
            "ATR_Buy": df["ATR"].iloc[0],
            "ATR_Sell": df["ATR"].iloc[-1],
            "Volume_Buy": df["Volume"].iloc[0],
            "Volume_Sell": df["Volume"].iloc[-1],
            "Stoch_K_Buy": df["Stoch_K"].iloc[0] if "Stoch_K" in df.columns else np.nan,
            "Stoch_K_Sell": df["Stoch_K"].iloc[-1] if "Stoch_K" in df.columns else np.nan,
            "Stoch_D_Buy": df["Stoch_D"].iloc[0] if "Stoch_D" in df.columns else np.nan,
            "Stoch_D_Sell": df["Stoch_D"].iloc[-1] if "Stoch_D" in df.columns else np.nan,
        }
        print("No ideal trade point found; using fallback trade point based on first and last observations.")
        return trade_features
    return None

# -------------------------------
# Main Simulation
# -------------------------------

# 1. Fetch real-time intraday data for the specified stock
print(f"Fetching real-time data for {stock_symbol}...")
data = fetch_intraday_data(stock_symbol, interval=interval, period=period)
if data is None or data.empty:
    raise Exception("No data fetched for the specified stock.")

# 2. Compute technical indicators
data = calculate_indicators(data)
data = calculate_additional_indicators(data)

# 3. Identify the best trade point (using the updated function)
trade_point = find_best_trade_point(data)
if trade_point is None:
    raise Exception("Could not determine a valid trade point from the data, even after fallback.")
    
buy_price = trade_point["Buy Price"]
actual_profit = trade_point["Profit"]  # For reference

print(f"Identified trade point for {stock_symbol}:")
print(f"  Buy Price: {buy_price}")
print(f"  Sell Price: {trade_point['Sell Price']}")
print(f"  Actual Profit (per share): {actual_profit}")

# 4. Prepare the features for prediction as expected by the model
feature_names = ["RSI_Buy", "RSI_Sell", "MACD_Buy", "MACD_Sell", 
                 "ATR_Buy", "ATR_Sell", "Volume_Buy", "Volume_Sell",
                 "Stoch_K_Buy", "Stoch_K_Sell", "Stoch_D_Buy", "Stoch_D_Sell"]

X_new = pd.DataFrame({fname: [trade_point[fname]] for fname in feature_names})

# 5. Use the model to predict the profit per share
predicted_profit_per_share = model.predict(X_new)[0]
print(f"Predicted Profit (per share) for {stock_symbol}: {predicted_profit_per_share:.4f}")

# 6. Calculate how many shares you can buy with your investment
shares_to_buy = math.floor(investment_amount / buy_price)
print(f"With an investment of ${investment_amount}, you can buy {shares_to_buy} shares at a buy price of ${buy_price:.2f} per share.")

# 7. Compute total predicted profit and final amount
total_predicted_profit = predicted_profit_per_share * shares_to_buy
final_value = investment_amount + total_predicted_profit

print("\n--- Investment Simulation ---")
print(f"Initial Investment: ${investment_amount:.2f}")
print(f"Number of Shares Bought: {shares_to_buy}")
print(f"Predicted Profit per Share: ${predicted_profit_per_share:.4f}")
print(f"Total Predicted Profit: ${total_predicted_profit:.2f}")
print(f"Final Estimated Value: ${final_value:.2f}")
