# import dependencies
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import yfinance as yf

# download AAPL stock data
start = "2015-01-01"
end = "2024-01-01"
stock = "AAPL"

data = yf.download(stock, start=start, end=end)

# Reset index and ensure it's a single-level index
data = data.reset_index()

# Check if the data columns are correct
# Print the first few rows of data to confirm
print(data.head())

# Initialize strategy class
class MySMAStrategy(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()

# Run the backtest
backtest = Backtest(data, MySMAStrategy, commission=.002, exclusive_orders=True)
stats = backtest.run()

print(stats)
