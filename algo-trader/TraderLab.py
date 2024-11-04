import json
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
import pandas as pd
from data_fetch import get_paper_creds
from preprocess import train_test_split

class TraderLab:

    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as file:
            config = json.load(file)

        # Set attributes
        for key, value in config.items():
            setattr(self, key, value)

    def fetch_data(self) -> None:
        """
        Fetches historical stock data for training and validation.

        This function pulls stock price and volume data for a list of symbols specified 
        in `train_symbols` config key. The data within is sampled with a period specified
        in the 'period' config key (in minutes)and a date range defined by `train_start_end`. 
        The retrieved data is then split into training and validation sets based on 
        `validation_split` and saved to `train_data` and `val_data` attributes.

        Parameters:
        None

        Returns:
        None
        """
        print("Pulling Training Data:")
        
        # Retrieve API credentials
        api_key, secret_key = get_paper_creds()
        
        # Initialize the client for fetching historical stock data
        stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Initialize dictionaries to store training and validation data for each symbol
        train_dict = {}
        val_dict = {}
        
        # Fetch data for each symbol in the training set
        for symbol in self.train_symbols:
            print(f"Fetching {symbol} data...")
            
            # Create request for stock bars with specified symbol, timeframe, and date range
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame(amount=self.period, unit=TimeFrameUnit.Minute),
                start=self.train_start_end[0],
                end=self.train_start_end[1]
            )
            
            # Retrieve data and select only 'close' and 'volume' columns
            df = stock_historical_data_client.get_stock_bars(req).df.loc[:, ["close", "volume"]]
            
            # Split data into training and validation sets
            train_dict[symbol], val_dict[symbol] = train_test_split(df, self.validation_split)
        
        # Store the split data into instance attributes
        self.train_data = train_dict
        self.val_data = val_dict


    def preprocess_data(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def backtest(self):
        pass

if __name__ == "__main__":
    pass