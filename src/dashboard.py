from price import StockData
import mplfinance as mpf
from movers import TopMovers

class Dashboard:
    
    """
        A class to generate a dashboard with stock price charts.
    """
    
    def __init__(self, symbols: list, period: str, interval: str):
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.stocks = self.get_stocks()
    
    def get_stocks(self) -> list:
        stocks = []
        for symbol in self.symbols:
            stock = StockData(symbol, self.period, self.interval)
            if stock.data is not None:
                stocks.append(stock)
        return stocks
    
    def generate_charts(self):
        for stock in self.stocks:
            mpf.plot(stock.data, type='candle', volume=True, style='charles', title=stock.symbol)


if __name__ == "__main__":
    symbols = TopMovers(100).symbols
    for symbol in symbols:
        # Generate a dashboard with stock price charts
        Dashboard([symbol], '1d', '5m').generate_charts()