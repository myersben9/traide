import yfinance
from typing import List, Dict, Union
import json
import pandas as pd

# Configuration constants
PERCENTAGE_CHANGE: float = 50
EXCHANGES: List[str] = ['NMS', 'NYQ','ASE', 'PCX', 'YHD', 'NCM']  # Supported exchanges

class TopMovers:
    """
    A class to fetch top movers from Yahoo Finance based on predefined screening
    criteria.

    Attributes:
        percentage_change (float): The minimum percentage change to filter stocks.
        symbols (List[str]): A list of stock symbols that meet the criteria.
        return_message (str): A message to return if no stocks are found.
        info (Dict[str, Union[str,int]]): A dictionary containing the screening info.
    """

    def __init__(self, percentage_change: float = PERCENTAGE_CHANGE):
        """
        Initialize the TopMovers class.

        Args:
            percentage_change (float): The minimum percentage change to filter stocks.
        """
        self.percentage_change = percentage_change
        self.screen = self.screen_percentages()
        self.quotes = self.screen['quotes'] if self.screen and 'quotes' in self.screen else []
        self.symbols = [quote['symbol'] for quote in self.quotes]

    def screen_percentages(self) -> List[str]:
        """
        Fetch stocks from Yahoo Finance based on predefined screening criteria.

        Returns:
            List[str]: A list of stock symbols that meet the criteria.
        """
        try:
            # Define the screening criteria
            q = yfinance.EquityQuery('and', [
                yfinance.EquityQuery('gt', ['percentchange', self.percentage_change]),  # Stocks with > percentage_change
                yfinance.EquityQuery('eq', ['region', 'us']),  # US region
                yfinance.EquityQuery('is-in', ['exchange', *EXCHANGES]),  # Specific exchanges
            ])

            # Fetch the screened stocks
            return yfinance.screen(q, sortField='percentchange', sortAsc=True, size=250)

        except Exception as e:
            print(f"Error fetching stocks: {e}")
            return []
# Example usage
if __name__ == "__main__":
    top_movers = TopMovers(PERCENTAGE_CHANGE)
    print(top_movers.symbols)  

