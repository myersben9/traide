import pandas as pd
import numpy as np
import yfinance as yf
import talib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
        best_trades = find_best_trade_points(stock_data)
        best_trades["Symbol"] = symbol
        all_trades.append(best_trades)

if all_trades:
    final_df = pd.concat(all_trades)
    final_df.to_csv(os.path.join(report_dir, "enhanced_stock_trades.csv"), index=False)
else:
    print("No trades found.")

# -------------------------------
# Statistical Analysis: Compute summary stats, skewness, kurtosis, and correlation matrix
# -------------------------------
analysis_cols = ["RSI_Buy", "RSI_Sell", "MACD_Buy", "MACD_Sell", 
                 "ATR_Buy", "ATR_Sell", "Volume_Buy", "Volume_Sell",
                 "Stoch_K_Buy", "Stoch_K_Sell", "Stoch_D_Buy", "Stoch_D_Sell", "Profit"]
stats_df = final_df[analysis_cols].dropna()
summary_stats = stats_df.describe()
skewness = stats_df.skew()
kurtosis = stats_df.kurt()
correlation = stats_df.corr()

summary_stats.to_csv(os.path.join(report_dir, "summary_statistics.csv"))
skewness.to_csv(os.path.join(report_dir, "skewness.csv"))
kurtosis.to_csv(os.path.join(report_dir, "kurtosis.csv"))
correlation.to_csv(os.path.join(report_dir, "correlation_matrix.csv"))

# -------------------------------
# Plotting Functions: KDE Distributions and Histograms
# -------------------------------
def plot_combined_distributions(df):
    indicators = ["RSI", "MACD", "ATR", "Volume", "Stoch_K", "Stoch_D"]
    plt.figure(figsize=(14, 12))
    for i, indicator in enumerate(indicators):
        plt.subplot(3, 2, i + 1)
        sns.kdeplot(df[f"{indicator}_Buy"], label=f"{indicator} at Buy", shade=True, color='blue')
        sns.kdeplot(df[f"{indicator}_Sell"], label=f"{indicator} at Sell", shade=True, color='red')
        plt.title(f"Probability Density of {indicator} at Buy & Sell Points")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "combined_distributions.png"))
    plt.show()

def plot_combined_histograms(df):
    indicators = ["RSI", "MACD", "ATR", "Volume", "Stoch_K", "Stoch_D"]
    plt.figure(figsize=(14, 12))
    for i, indicator in enumerate(indicators):
        plt.subplot(3, 2, i + 1)
        sns.histplot(df[f"{indicator}_Buy"], label=f"{indicator} at Buy", bins=30, kde=True, color='blue', alpha=0.6)
        sns.histplot(df[f"{indicator}_Sell"], label=f"{indicator} at Sell", bins=30, kde=True, color='red', alpha=0.6)
        plt.title(f"Histogram of {indicator} at Buy & Sell Points")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "combined_histograms.png"))
    plt.show()

plot_combined_distributions(final_df)
plot_combined_histograms(final_df)

# -------------------------------
# Regression Modeling for Statistical Validation
# -------------------------------

# Define predictor variables and target variable
predictors = ["RSI_Buy", "RSI_Sell", "MACD_Buy", "MACD_Sell", 
              "ATR_Buy", "ATR_Sell", "Volume_Buy", "Volume_Sell",
              "Stoch_K_Buy", "Stoch_K_Sell", "Stoch_D_Buy", "Stoch_D_Sell"]
data_reg = final_df[predictors + ["Profit"]].dropna()

# 1. OLS Regression using statsmodels
X = data_reg[predictors]
y = data_reg["Profit"]
X_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_const).fit()
print(model_sm.summary())
with open(os.path.join(report_dir, "regression_summary.txt"), "w") as f:
    f.write(model_sm.summary().as_text())

# 2. Linear Regression using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
r_squared_lr = model_lr.score(X_test, y_test)
print("Scikit-learn Linear Regression R-squared:", r_squared_lr)

# 3. Ridge Regression (to address multicollinearity) using cross-validation
alphas = [0.1, 1.0, 10.0, 100.0]
# Create a pipeline with standard scaling and ridge regression
ridge_pipeline = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, scoring="r2"))
ridge_pipeline.fit(X_train, y_train)
r_squared_ridge = ridge_pipeline.score(X_test, y_test)
ridge_coefficients = ridge_pipeline.named_steps['ridgecv'].coef_
print("Ridge Regression R-squared:", r_squared_ridge)
print("Ridge Regression Coefficients:", dict(zip(predictors, ridge_coefficients)))

# Save Ridge regression results
ridge_results = pd.DataFrame({"Coefficient": ridge_coefficients}, index=predictors)
ridge_results.to_csv(os.path.join(report_dir, "ridge_regression_results.csv"))

# -------------------------------
# Real-Time Data Testing
# -------------------------------
def fetch_real_time_data(symbol, interval="1m", period="1d"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(interval=interval, period=period)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data['Symbol'] = symbol
        return data
    except Exception as e:
        print(f"Error fetching real-time data for {symbol}: {e}")
        return None

def apply_model_to_realtime(model, symbol):
    data_rt = fetch_real_time_data(symbol)
    if data_rt is None or data_rt.empty:
        print("No real-time data available for", symbol)
        return None
    data_rt = calculate_indicators(data_rt)
    data_rt = calculate_additional_indicators(data_rt)
    # Use the latest data point for prediction
    latest = data_rt.iloc[-1:]
    realtime_features = pd.DataFrame({
        "RSI_Buy": latest["RSI"],
        "RSI_Sell": latest["RSI"],
        "MACD_Buy": latest["MACD"],
        "MACD_Sell": latest["MACD"],
        "ATR_Buy": latest["ATR"],
        "ATR_Sell": latest["ATR"],
        "Volume_Buy": latest["Volume"],
        "Volume_Sell": latest["Volume"],
        "Stoch_K_Buy": latest["Stoch_K"],
        "Stoch_K_Sell": latest["Stoch_K"],
        "Stoch_D_Buy": latest["Stoch_D"],
        "Stoch_D_Sell": latest["Stoch_D"],
    })
    # Predict profit using the scikit-learn linear model (model_lr) as an example
    predicted_profit = model.predict(realtime_features)[0]
    print(f"Predicted Profit for {symbol} using real-time data: {predicted_profit}")
    return predicted_profit

# Example: Apply models to real-time data for AAPL
apply_model_to_realtime(model_lr, "AAPL")
apply_model_to_realtime(ridge_pipeline, "AAPL")

print("Analysis, regression modeling, and real-time testing complete. Results have been saved in the 'data_report' folder.")
