import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.movers import TopMovers

# TICKER ='MGOL'

# interval = '1m' 
# period = '1d'

# ticker = yf.Ticker(TICKER)
# # Get stock average
# stock_info = ticker.info
# averageVolume10Day = stock_info["averageDailyVolume10Day"]
# averageVolume = stock_info["averageVolume"]

# download = yf.download(TICKER, interval=interval, period=period, progress=False)


# data = pd.DataFrame(download)

# data.dropna(inplace=True)

# fig, ax = plt.subplots(2,sharex=True, figsize=(12,8))

# fig.suptitle('Volume and Price')

# # Generate buy signals if the volume is greater than the average volume
# volume = data['Volume'].to_numpy().flatten()
# close = data['Close'].to_numpy().flatten()
# buySignal = {}
# sellSignal = {data.index[-1]: close[-1]}
# for i in range(len(data)):
#     volume = volume[i]
#     if volume > averageVolume:
#         # Generate buy signal
#         buySignal[data.index[i]] = close[i]
#         break
# # Plot the buy and sell signals
# ax[0].scatter(buySignal.keys(), buySignal.values(), color='green', label='Buy Signal', marker='^')
# ax[0].scatter(sellSignal.keys(), sellSignal.values(), color='red', label='Sell Signal', marker='v')
# ax[0].plot(data.index, data['Close'], label='Close Price', color='black')
# ax[0].legend()
# ax[0].xaxis.set_tick_params(labelbottom=True)

# data.index = pd.to_datetime(data.index).tz_convert('US/Pacific')

# x = data.index.to_numpy().flatten()

# # Get another bar to the graph called moving average volume
# y2 = data["Volume"].rolling(window=10).mean().to_numpy().flatten()
# y1 = data["Volume"].to_numpy().flatten()


# categories = ['Volume', 'Moving Average Volume']

# width =0.001
# # Plot the change in volume instead of the volume

# ax[1].bar(x,y1, label='Volume', color='blue', width = width)
# # Plot two horizont lines for average volume and average volume 10 days
# ax[1].axhline(averageVolume, color='red', linewidth=1, linestyle='--', label='Average Volume')
# ax[1].axhline(averageVolume10Day, color='blue', linewidth=1, linestyle='--', label='Average Volume 10 Days')
# # Create a legend
# ax[1].legend()


# # Calculate price per share. Close price of the sell signal minus the close price of the buy signal
# # pricePerShare = sellSignal[list(sellSignal.keys())[0]] - buySignal[list(buySignal.keys())[0]]
# # print(f'Profit per share: {pricePerShare}')


# plt.show()


class Backtest:
    """
    Class to backtest momentum strategy of volume exceeding average 10 day volume. 
    It buys when that happens and sells at end of the day.

    Args:
        data (pd.DataFrame): The stock data to backtest.
    """
    def __init__(self, data: pd.DataFrame, symbol: str):
        self.data = data
        self.close = data['Close'].to_numpy().flatten()
        self.volume = data['Volume'].to_numpy().flatten()
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.averageVolume10Day = self.ticker.info["averageVolume10days"]
        self.averageVolume = self.ticker.info["averageVolume"]
        self.buySignal = {}
        self.sellSignal = {self.data.index[-1]: self.close[-1]}
        self.profit_per_share = self.backtest()
        self.capital = 500
        self.shares = 0
        self.buy()
        self.sell()
        

    def backtest(self):
        """
        Backtest the momentum strategy.
        """
        print(self.ticker.info)
        try:
            for i in range(len(self.data)):

                volume = self.volume[i]
                if volume > self.averageVolume:
                    # Generate buy signal
                    self.buySignal[self.data.index[i]] = self.close[i]
                    break
                else:
                    raise ValueError("Volume is not greater than average volume")
            # Calculate price per share
            return self.sellSignal[list(self.sellSignal.keys())[0]] - self.buySignal[list(self.buySignal.keys())[0]]
        except ValueError as e:
            print(f"Error backtesting: {e}")
            return 0
    
    def buy(self):
        """
        Buy the maxium amount of shares possible at the buy signal with the current capital.
        """
        self.shares = self.capital // self.buySignal[list(self.buySignal.keys())[0]]
        self.capital -= self.shares * self.buySignal[list(self.buySignal.keys())[0]]

    def sell(self):
        """
        Sell all shares at the sell signal.
        """
        self.capital += self.shares * self.sellSignal[list(self.sellSignal.keys())[0]]
        self.shares = 0
    
    def get_total_profit(self):
        """
        Get the total profit.

        Returns:
            float: The total profit.
        """
        return self.capital * self.profit_per_share

# Create a backtest object
if __name__ == "__main__":  
    TICKER ='PINS'

    interval = '1m' 
    period = '1d'

    ticker = yf.Ticker(TICKER)
    download = yf.download(TICKER, interval=interval, period=period, progress=False)


    data = pd.DataFrame(download)

    data.dropna(inplace=True)

    backtest = Backtest(data, TICKER)
    print("profit",backtest.profit_per_share)
    print("total profit", backtest.get_total_profit())


# print(symbols)






        