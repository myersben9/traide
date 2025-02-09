import pandas as pd
import numpy as np
import yfinance as yf
import talib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# -------------------------------
# Create directory for outputs
# -------------------------------
report_dir = "data_report"
os.makedirs(report_dir, exist_ok=True)

# -------------------------------
# Load stock symbols from CSV (NASDAQ list)
# -------------------------------
csv_file = "nasdaq.csv"  # Replace with your file path
df = pd.read_csv(csv_file)
sample_stocks = df.sample(n=500, random_state=42)['Symbol'].tolist()

# -------------------------------
# Function: Fetch intraday data from Yahoo Finance
# -------------------------------
def fetch_intraday_data(symbol, interval="5m", period="5d"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(interval=interval, period=period)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data['Symbol'] = symbol
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# -------------------------------
# Function: Calculate basic technical indicators (RSI, MACD, ATR)
# -------------------------------
def calculate_indicators(df):
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    df["MACD"], df["MACD Signal"], _ = talib.MACD(df["Close"])
    df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    return df

# -------------------------------
# Function: Calculate additional indicators (Bollinger Bands, Stochastic Oscillator)
# -------------------------------
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
# Function: Identify best trade points per day and compute Profit
# -------------------------------
def find_best_trade_points(df):
    df["Date"] = df.index.date
    best_trades = []
    for date, group in df.groupby("Date"):
        min_price = group["Low"].min()
        max_price = group["High"].max()
        min_time = group[group["Low"] == min_price].index[0]
        max_time = group[group["High"] == max_price].index[0]
        if min_time < max_time:
            best_trades.append({
                "Date": date,
                "Buy Time": min_time,
                "Buy Price": min_price,
                "Sell Time": max_time,
                "Sell Price": max_price,
                "Profit": max_price - min_price,  # Compute Profit as difference
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
                "UpperBand_Buy": group.loc[min_time, "UpperBand"] if "UpperBand" in group.columns else np.nan,
                "MiddleBand_Buy": group.loc[min_time, "MiddleBand"] if "MiddleBand" in group.columns else np.nan,
                "LowerBand_Buy": group.loc[min_time, "LowerBand"] if "LowerBand" in group.columns else np.nan,
                "UpperBand_Sell": group.loc[max_time, "UpperBand"] if "UpperBand" in group.columns else np.nan,
                "MiddleBand_Sell": group.loc[max_time, "MiddleBand"] if "MiddleBand" in group.columns else np.nan,
                "LowerBand_Sell": group.loc[max_time, "LowerBand"] if "LowerBand" in group.columns else np.nan,
            })
    return pd.DataFrame(best_trades)

# -------------------------------
# Process each stock: Fetch data, compute indicators, and extract trade points
# -------------------------------
all_trades = []
for symbol in sample_stocks:
    print(f"Processing {symbol}...")
    stock_data = fetch_intraday_data(symbol)
    if stock_data is not None and not stock_data.empty:
        stock_data = calculate_indicators(stock_data)
        stock_data = calculate_additional_indicators(stock_data)
        trade_points = find_best_trade_points(stock_data)
        if not trade_points.empty:
            trade_points["Symbol"] = symbol
            all_trades.append(trade_points)

if all_trades:
    final_df = pd.concat(all_trades, ignore_index=True)
    final_df.to_csv(os.path.join(report_dir, "enhanced_stock_trades.csv"), index=False)
else:
    print("No trades found.")

# -------------------------------
# Regression Modeling for Profit Prediction
# -------------------------------
predictors = ["RSI_Buy", "RSI_Sell", "MACD_Buy", "MACD_Sell", 
              "ATR_Buy", "ATR_Sell", "Volume_Buy", "Volume_Sell",
              "Stoch_K_Buy", "Stoch_K_Sell", "Stoch_D_Buy", "Stoch_D_Sell"]
target = "Profit"

reg_data = final_df[predictors + [target]].dropna()

X = reg_data[predictors]
y = reg_data[target]

# Train a Linear Regression model on the entire dataset
model_lr = LinearRegression()
model_lr.fit(X, y)

# Evaluate performance on the training data
r_squared = model_lr.score(X, y)
print("Final Linear Regression R-squared on 500 stocks:", r_squared)

# Save the trained model using joblib
model_path = os.path.join(report_dir, "profit_predictor_model.pkl")
joblib.dump(model_lr, model_path)
print(f"Model saved to {model_path}")

# Save regression summary statistics using statsmodels for detailed summary
X_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_const).fit()
with open(os.path.join(report_dir, "final_regression_summary.txt"), "w") as f:
    f.write(model_sm.summary().as_text())

# Save summary statistics and correlation matrix as CSV files
reg_data.describe().to_csv(os.path.join(report_dir, "regression_data_summary.csv"))
reg_data.corr().to_csv(os.path.join(report_dir, "regression_data_correlation.csv"))

print("Training complete. Regression model and outputs are saved in the 'data_report' folder.")
