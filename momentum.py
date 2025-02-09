import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Download stock data
data = yf.download('LTRY', interval='1m', period='1d')



# Calculate momentum
data['Momentum'] = data['Close'].pct_change(periods=10)

# Generate signals
data['Signal'] = np.where(data['Momentum'] > 0, 1.0, -1.0)
data['Position'] = data['Signal'].diff()

# Plot momentum using matplotlib
plt.figure(figsize=(14, 7))

# Add Momentum trace
plt.plot(data.index, data['Momentum'], label='Momentum', color='blue')

# Add zero line for reference
plt.axhline(y=0, linestyle='--', color='black')

# Update layout
plt.title('Momentum Strategy for AAPL')
plt.xlabel('Date')
plt.ylabel('Momentum')
plt.legend(loc='best')
plt.grid(True)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# Show the plot
plt.tight_layout()
plt.show()