import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from movers import TopMovers
import mplfinance as mpf
import pytz  # For timezone handling

class StockData:

    """
        A class to fetch stock data and generate price charts.
    """

    def __init__(self, symbol: str, period: str, interval: str):
        self.symbol = symbol
        self.period =  period
        self.interval = interval
        self.data = self.get_data()

    def get_data(self) -> pd.DataFrame:
        try:
            stock = yf.Ticker(self.symbol)
            data = stock.history(period=self.period, interval=self.interval)
            # Turn data into a DataFrame
            data = pd.DataFrame(data)
            # Drop any NaN values
            data = data.dropna()
            # Conver to PST timezone
            print(data)
            return data
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None

if __name__ == "__main__":
    top_movers = TopMovers(80)
    print(top_movers.symbols)

