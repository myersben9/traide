import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from movers import TopMovers
import mplfinance as mpf
import pytz  # For timezone handling

class Graphs:

    """
        A class to fetch stock data and generate price charts.
    """

    def __init__(self, symbol: str, period: str, interval: str):
        self.symbol = symbol
        self.period =  period
        self.interval = interval
        self.data = self.get_data()
        self.plot_price()

    def get_data(self) -> pd.DataFrame:
        try:
            data = yf.Ticker(self.symbol)
            data = data.history(period=self.period, interval=self.interval)

            # Conver to PST timezone
            data.index = pd.DataFrame(data.index).tz_convert(pytz.timezone('US/Pacific'))


            if data.empty:
                print(f"No data available for {self.symbol}")
                return None
            return data
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None

    def plot_price(self) -> None:
        try:
            daily = self.data
            # Check if the daily data is less than 14 rows
            if daily is not None and len(daily) < 14:
                print(f"Insufficient data for {self.symbol}. Skipping chart generation.")
                return
            
            # Plot the data with the x-axis in PST
            mpf.plot(daily, type='candle', mav=(3,6,9), volume=True, show_nontrading=True, 
                    style='yahoo', title=self.symbol, 
                    ylabel='Price', ylabel_lower='Volume', savefig=f'charts/{self.symbol}.png')
            
        except Exception as e:
            print(f"Error plotting price for {self.symbol}: {e}")

if __name__ == "__main__":
    top_movers = TopMovers(80)
    for symbol in top_movers.symbols:
        Graphs(symbol, period='1d', interval='5m')