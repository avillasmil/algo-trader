import json
from datetime import datetime, timedelta
from alpaca.data.historical.stock import StockHistoricalDataClient
import pandas as pd
import multiprocessing as mp
from functools import partial
from data_fetch import get_paper_creds, fetch_symbol_data
from preprocess import (
    train_test_split,
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_obv,
    calculate_macd,
    generate_labels,
    generate_label_summary
)
from train import split_features_and_labels
from xgboost import XGBClassifier


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
        Parallel fetch of training and validation data for multiple stock symbols.
        """
        print("Pulling Training Data:")
        
        # Retrieve API credentials and initialize the stock data client
        api_key, secret_key = get_paper_creds()
        stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Prepare partial function to pass additional parameters to `fetch_symbol_data`
        fetch_func = partial(
            fetch_symbol_data,
            period=self.period,
            train_start_end=self.train_start_end,
            validation_split=self.validation_split,
            stock_historical_data_client=stock_historical_data_client
        )
        
        # Use multiprocessing pool to fetch data in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(fetch_func, self.train_symbols)
        
        # Collect results into dictionaries
        self.train_data = {symbol: train_data for symbol, train_data, _ in results}
        self.val_data = {symbol: val_data for symbol, _, val_data in results}


    def preprocess_data(self) -> None:
        """
        Preprocesses training and validation data by calculating technical indicators 
        (Bollinger Bands, SMA, RSI, OBV, EMA, MACD) for each stock symbol, and generates 
        labels based on shifted data.

        This method:
        1. Calculates specified indicators and adds them as new columns.
        2. Generates target labels for each symbol. (0 = No Action, 1 = Buy, 2 = Sell)
        3. Combines all symbols' dataframes into a single training and validation dataframe.
        4. Drops any NaN values resulting from indicator calculations.

        """
        print("Preprocessing Data...")
        # Calculate indicators and generate labels for each dataset (train and validation)
        for data in (self.train_data, self.val_data):
            for symbol in self.train_symbols:
                # Access the symbol data and compute indicators
                symbol_data = data[symbol]
                symbol_data['bb'] = calculate_bollinger_bands(symbol_data['close'], window=self.feat_win_length)
                symbol_data['sma'] = calculate_sma(symbol_data['close'], window=self.feat_win_length)
                symbol_data['rsi'] = calculate_rsi(symbol_data['close'], window=self.feat_win_length)
                symbol_data['obv'] = calculate_obv(symbol_data)
                symbol_data['ema'] = calculate_ema(symbol_data['close'], window=self.feat_win_length)
                
                # Calculate MACD and concatenate
                macd_df = calculate_macd(symbol_data['close'])
                symbol_data = pd.concat([symbol_data, macd_df], axis=1)
                
                # Generate and assign labels
                labels_df = symbol_data.shift(periods=-2)
                symbol_data['label'] = generate_labels(labels_df)
                
                # Update the dictionary with the processed data
                data[symbol] = symbol_data

        # Concatenate all symbols' data for train and validation sets, respectively
        self.train_data = pd.concat(self.train_data.values(), ignore_index=True).dropna()
        self.val_data = pd.concat(self.val_data.values(), ignore_index=True).dropna()

        # Label Overview
        generate_label_summary(self.train_data, self.val_data)
        

    def train(self):
        if self.model_to_use == "xgb":
            model = XGBClassifier()

        model.set_params(**self.model_params)

        X_train, y_train = split_features_and_labels(self.train_data)
        X_test, y_test = split_features_and_labels(self.val_data)

        if self.label_weights_flag:
            class_weights = y_train.value_counts(normalize=True)  # Get class distribution
            total_samples = len(y_train)
            scale_pos_weight = total_samples / (len(class_weights) * class_weights)

        model.fit(X_train, y_train, sample_weight=y_train.map(scale_pos_weight))

        self.model = model


    def evaluate(self):
        pass

    def backtest(self):
        pass

if __name__ == "__main__":
    pass