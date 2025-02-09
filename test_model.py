import pandas as pd
import numpy as np
import yfinance as yf
import talib
import os
import joblib
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Set up output directory for test results
# -------------------------------
test_output_dir = "test_model_report"
os.makedirs(test_output_dir, exist_ok=True)

# -------------------------------
# Load the saved model from training
# -------------------------------
model_path = os.path.join("data_report", "profit_predictor_model.pkl")
model_lr = joblib.load(model_path)
print(f"Loaded model from {model_path}")

# -------------------------------
# Load stock symbols from CSV (NASDAQ list) and sample a new set
# -------------------------------
csv_file = "nasdaq.csv"  # Replace with your file path
df = pd.read_csv(csv_file)
# Use a different random seed to get a new sample (e.g., 50 stocks)
sample_stocks = df.sample(n=50, random_state=123)['Symbol'].tolist()

# -------------------------------
# Define functions (same as in training)
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
            })
    return pd.DataFrame(best_trades)

# -------------------------------
# Process each stock in the test sample
# -------------------------------
all_trades = []
for symbol in sample_stocks:
    print(f"Processing {symbol} for testing...")
    data = fetch_intraday_data(symbol)
    if data is not None and not data.empty:
        data = calculate_indicators(data)
        data = calculate_additional_indicators(data)
        trade_points = find_best_trade_points(data)
        if not trade_points.empty:
            trade_points["Symbol"] = symbol
            all_trades.append(trade_points)

if all_trades:
    test_df = pd.concat(all_trades, ignore_index=True)
    test_df.to_csv(os.path.join(test_output_dir, "test_enhanced_stock_trades.csv"), index=False)
else:
    print("No trades found in the test sample.")
    exit()

# -------------------------------
# Regression Evaluation on Test Data
# -------------------------------
predictors = ["RSI_Buy", "RSI_Sell", "MACD_Buy", "MACD_Sell", 
              "ATR_Buy", "ATR_Sell", "Volume_Buy", "Volume_Sell",
              "Stoch_K_Buy", "Stoch_K_Sell", "Stoch_D_Buy", "Stoch_D_Sell"]
target = "Profit"

test_data = test_df[predictors + [target]].dropna()
X_test_data = test_data[predictors]
y_test_data = test_data[target]

# Use the loaded model to predict Profit for the test data
y_pred = model_lr.predict(X_test_data)

# Compute performance metrics
r_squared_test = r2_score(y_test_data, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test_data, y_pred))
mae_test = np.mean(np.abs(y_test_data - y_pred))

# Generate a statistical report for the test sample
report_lines = [
    "Statistical Report for Test Model on New Sample Set",
    "-----------------------------------------------------",
    f"Number of Observations: {len(test_data)}",
    f"R-squared: {r_squared_test:.3f}",
    f"RMSE: {rmse_test:.3f}",
    f"MAE: {mae_test:.3f}",
    "",
    "Regression Predictions Summary:",
]

summary_df = test_data.copy()
summary_df["Predicted Profit"] = y_pred
report_lines.append(summary_df.describe().to_string())

report_text = "\n".join(report_lines)
with open(os.path.join(test_output_dir, "test_statistical_report.txt"), "w") as f:
    f.write(report_text)

print(report_text)
print("Test model evaluation complete. Reports and outputs are saved in the 'test_model_report' folder.")
