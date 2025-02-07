import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Download data
data = yf.download('LTRY', interval='1m', period='2d')
data['Fast_MA'] = data['Close'].rolling(window=10).mean()
data['Slow_MA'] = data['Close'].rolling(window=50).mean()

# Generate signals
data['Signal'] = 0.0
data['Signal'] = np.where(data['Fast_MA'] > data['Slow_MA'], 1.0, -1.0)
data['Position'] = data['Signal'].diff()

# Plotting
plt.figure(figsize=(14, 7))

# Plot Close price
plt.plot(data.index, data['Close'], label='AAPL Close', color='blue', alpha=0.5)

# Plot Fast Moving Average
plt.plot(data.index, data['Fast_MA'], label='10-Day MA', color='red', alpha=0.75)

# Plot Slow Moving Average
plt.plot(data.index, data['Slow_MA'], label='50-Day MA', color='green', alpha=0.75)

# Plot Buy signals
plt.plot(data[data['Position'] == 2].index, data['Fast_MA'][data['Position'] == 2], '^', markersize=10, color='gold', label='Buy Signal')

# Plot Sell signals
plt.plot(data[data['Position'] == -2].index, data['Fast_MA'][data['Position'] == -2], 'v', markersize=10, color='black', label='Sell Signal')

# Formatting the plot
plt.title('Moving Average Crossover Strategy with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='best')
plt.grid(True)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

# Show the plot
plt.tight_layout()
plt.show()