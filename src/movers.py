import yfinance
from typing import List, Dict, Union
import json
import pandas as pd

# Configuration constants
PERCENTAGE_CHANGE: float = 68
EXCHANGES: List[str] = ["NMS", "NGM", "NCM", "NYQ", "ASE", "PCX"]  # Supported exchanges

class TopMovers:
    """
    A class to fetch top movers from Yahoo Finance based on predefined screening
    criteria. Does not contain permarket movers.

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
        self.response = self.top_movers_payload()

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
            screen = yfinance.screen(q, sortField='percentchange', sortAsc=False, size=250)


            screen["quotes"] = [quote for quote in screen["quotes"] if 'firstTradeDateMilliseconds' in quote and quote['quoteSourceName'] != 'Delayed Quote']

            return screen

        except Exception as e:
            print(f"Error fetching stocks: {e}")
            return []
        
    # Make a Top Movers payload for the API with the top movers and there history

    def top_movers_payload(self) -> Dict:
        """
        Create a payload for the API containing the top movers and their historical data.

        Returns:
            Dict: A dictionary containing the top movers and their historical data.
        """
        tickers = yfinance.Tickers(self.symbols)
        history = tickers.history(period="1d", interval="1m", group_by='ticker')
        # Make the payload
        payload = {}       
        for ticker in self.symbols:
            payload[ticker] = pd.DataFrame(history[ticker]).to_dict(orient='records')
 
        return payload


        
# Example usage
if __name__ == "__main__":
    top_movers = TopMovers(PERCENTAGE_CHANGE)
    print(top_movers.symbols)
    top_movers.top_movers_payload()

    

