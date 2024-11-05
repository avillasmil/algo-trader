import json
import pickle
import os
from datetime import datetime, timedelta
from alpaca.data.historical.stock import StockHistoricalDataClient
import pandas as pd
import multiprocessing as mp
from functools import partial
from data_fetch import get_paper_creds, fetch_symbol_data, fetch_symbol_data_BT
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
from backtest import generate_BT_features, preprocess_BT_data, run_simulation
from train import split_features_and_labels, get_model_filename, get_sample_weights
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class TraderLab:

    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as file:
            config = json.load(file)

        # Set attributes
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)

        # Retrieve API credentials and initialize the stock data client
        api_key, secret_key = get_paper_creds()
        self.stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key)


    def fetch_data(self) -> None:
        """
        Parallel fetch of training and validation data for multiple stock symbols.
        """
        print("Pulling Training Data:")
        
        # Prepare partial function to pass additional parameters to `fetch_symbol_data`
        fetch_func = partial(
            fetch_symbol_data,
            period=self.period,
            train_start_end=self.train_start_end,
            validation_split=self.validation_split,
            stock_historical_data_client=self.stock_historical_data_client
        )
        
        # Use multiprocessing pool to fetch data in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(fetch_func, self.train_symbols)
        
        # Collect results into dictionaries
        self.train_data = {symbol: train_data for symbol, train_data, _ in results}
        self.val_data = {symbol: val_data for symbol, _, val_data in results}
        print("--------------------------------------------------------------------------")


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
        print("Preprocessing Data:")
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
                labels_df = symbol_data.shift(periods=-self.lookahead_periods)
                symbol_data['label'] = generate_labels(labels_df)
                
                # Update the dictionary with the processed data
                data[symbol] = symbol_data

        # Concatenate all symbols' data for train and validation sets, respectively
        self.train_data = pd.concat(self.train_data.values(), ignore_index=True).dropna()
        self.val_data = pd.concat(self.val_data.values(), ignore_index=True).dropna()

        # Label Overview
        generate_label_summary(self.train_data, self.val_data)
        print("--------------------------------------------------------------------------")

        

    def train(self) -> None:
        """
        Trains a specified machine learning model (currently supports XGBoost) with parameters 
        defined in `self.model_params`. The function applies class weighting if `self.label_weights_flag` 
        is set, splits data into features and labels, and then fits the model. After training, 
        it saves the trained model to a unique file using a configuration-based naming convention.

        """
        print("Training Model:")
        # Initialize the model based on selection
        if self.model_to_use == "xgb":
            model = XGBClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model: {self.model_to_use}")

        # Set model parameters from `self.model_params`
        model.set_params(**self.model_params)

        # Split train data into features and labels
        X_train, y_train = split_features_and_labels(self.train_data)

        # Calculate class weights if specified
        sample_weight = None
        if self.label_weights_flag:
            sample_weight = get_sample_weights(y_train)

        # Fit the model, using sample weights if they are calculated
        model.fit(X_train, y_train, sample_weight=sample_weight)
        self.model = model

        # Serialize and save the model with a unique filename
        path = get_model_filename(self.config)
        # Create the parent directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Training Complete. Model saved: {path}")
        print("--------------------------------------------------------------------------")


    def evaluate(self) -> None:
        print("Model Evaluation on Validation Set:")
        X_test, y_test = split_features_and_labels(self.val_data)
        y_preds = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_preds)
        print(f"Accuracy: {accuracy:.2f}")

        # Calculate precision, recall, and F1 score (averaging by 'macro', 'micro', or 'weighted')
        precision = precision_score(y_test, y_preds, average='weighted')
        recall = recall_score(y_test, y_preds, average='weighted')
        f1 = f1_score(y_test, y_preds, average='weighted')

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_preds)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        # Classification Report
        class_report = classification_report(y_test, y_preds)
        print("\nClassification Report:")
        print(class_report)
        print("--------------------------------------------------------------------------")


    def backtest(self):
        """
        Conducts a backtest simulation for specified stock symbols by fetching historical data, 
        generating features, preprocessing data, and running a simulation to assess portfolio performance.

        Steps:
        1. Fetch historical data in parallel for each symbol.
        2. Generate and preprocess technical indicators/features.
        3. Initialize portfolio with cash allocations and zeroed positions.
        4. Run a trading simulation based on model predictions, tracking daily portfolio values.

        """
        
        # Prepare a partial function to pass additional arguments to `fetch_symbol_data_BT`
        fetch_func = partial(
            fetch_symbol_data_BT,
            period=self.period,
            backtest_start_end=self.backtest_start_end,
            stock_historical_data_client=self.stock_historical_data_client
        )

        # Fetch historical data in parallel using multiprocessing
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(fetch_func, self.backtest_symbols)
        
        # Compile fetched data into a dictionary keyed by symbol
        self.backtest_data = {symbol: df for symbol, df in results}
        
        # Generate and preprocess features for each symbol's data
        self.backtest_data = generate_BT_features(self.backtest_data, self.backtest_symbols)
        self.backtest_data = preprocess_BT_data(self.backtest_data)

        # Initialize portfolio with specified cash allocations and empty positions
        portfolio = {
            'cash': {symbol: self.initial_cash * self.allocations[symbol] for symbol in self.backtest_symbols},
            'positions': {symbol: {'shares': 0, 'value': 0} for symbol in self.backtest_symbols},
            'history': []  # To record daily portfolio values over the backtesting period
        }

        # Run the trading simulation, updating portfolio based on model predictions
        portfolio = run_simulation(self.backtest_data, self.backtest_symbols, self.model, portfolio, self.initial_cash)
        


if __name__ == "__main__":
    pass