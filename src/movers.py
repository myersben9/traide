import yfinance
from typing import List, Dict, Union
import json
import pandas as pd

# Configuration constants
PERCENTAGE_CHANGE: float = 80
EXCHANGES: List[str] = ["NMS", "NGM", "NCM", "NYQ", "ASE", "PCX"]  # Supported exchanges

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
            screen = yfinance.screen(q, sortField='percentchange', sortAsc=True, size=250)

            # Put quotes in a csv file
            quotes = screen['quotes']
            # Remove quotes that don't have a firstTradeDateMilliseconds
            quotes = [quote for quote in quotes if 'firstTradeDateMilliseconds' in quote]

            # Pull stock history of the 

            quotes_df = pd.DataFrame(quotes)

            print(quotes_df.head())
            # Save the quotes to a csv file
            #
            quotes_df.to_csv('data/top_movers.csv', index=False)



            return screen

        except Exception as e:
            print(f"Error fetching stocks: {e}")
            return []


# Example usage
if __name__ == "__main__":
    top_movers = TopMovers(PERCENTAGE_CHANGE)
    with open('data/top_movers.json', 'w') as f:
        json.dump(top_movers.screen, f, indent=4)

