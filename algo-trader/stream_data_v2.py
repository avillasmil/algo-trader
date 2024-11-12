import asyncio
import pickle
import logging
from alpaca.data.live import StockDataStream
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict, deque
from data_fetch import get_paper_creds
from preprocess import (
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_obv,
    calculate_macd
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Alpaca API credentials
API_KEY, SECRET_KEY = get_paper_creds()
BASE_URL = 'https://paper-api.alpaca.markets'

# Symbols to subscribe to
symbols = ["MSFT", "AAPL", "GOOG"]

# Initialize WebSocket client
wss_client = StockDataStream(API_KEY, SECRET_KEY)

# Variables for 5-minute look-ahead aggregation
interval = timedelta(minutes=5)

# Load trained classifier model (assuming the model is already trained and saved)
with open("models/model_xgb_e67d3402.pkl", "rb") as f:
    classifier = pickle.load(f)

# Manage symbol-specific state with a class
class SymbolProcessor:
    def __init__(self, symbol, buffer_size = 26):
        self.symbol = symbol
        self.buffer_size = buffer_size
        self.next_aggregation_time = None
        self.interval_data = []
        self.rolling_window = deque(maxlen=buffer_size)
        self.obv_value = 0
        self.aggregated_data = []

    def initialize_aggregation_time(self, current_time):
        self.next_aggregation_time = current_time.replace(second=0, microsecond=0)
        if self.next_aggregation_time.minute % 5 != 0:
            self.next_aggregation_time += timedelta(minutes=(5 - self.next_aggregation_time.minute % 5))

    async def process_bar(self, bar, current_time):
        self.interval_data.append(bar)

        # Initialize aggregation time if not already set
        if self.next_aggregation_time is None:
            self.initialize_aggregation_time(current_time)

        # Check if current time has reached the 5-minute interval
        if current_time >= self.next_aggregation_time + interval:
            if self.interval_data:
                # Aggregate close price and volume
                close_price = self.interval_data[-1].close
                total_volume = sum(item.volume for item in self.interval_data)

                new_row = {"timestamp": self.next_aggregation_time, "close": close_price, "volume": total_volume}
                
                # Update OBV only based on the latest close and volume values
                if len(self.rolling_window) > 0:
                    last_close = self.rolling_window[-1]["close"]
                    if close_price > last_close:
                        self.obv_value += total_volume
                    elif close_price < last_close:
                        self.obv_value -= total_volume

                new_row["obv"] = self.obv_value
                
                self.rolling_window.append(new_row)
                self.aggregated_data.append(new_row)

                rolling_window_df = pd.DataFrame(list(self.rolling_window))
                # Calculate technical indicators
                if len(self.rolling_window) == self.buffer_size:
                    rolling_window_df["bb"] = calculate_bollinger_bands(rolling_window_df["close"], window = 2)
                    rolling_window_df["rsi"] = calculate_rsi(rolling_window_df["close"], window = 2)
                    rolling_window_df["sma"] = calculate_sma(rolling_window_df["close"], window = 2)
                    rolling_window_df["ema"] = calculate_ema(rolling_window_df["close"], window = 2)
                    macd_df = calculate_macd(rolling_window_df["close"], short_window=1, long_window=3, signal_window=2)
                    rolling_window_df = pd.concat([rolling_window_df, macd_df], axis=1)

                    # Extract the last row as feature input for classifier
                    feature_row = rolling_window_df.iloc[-1][["close", "volume", "bb", "rsi", "sma", "ema", "obv", "MACD", "Signal", "Histogram"]]
                    features = feature_row.values.reshape(1, -1)  # Reshape for a single prediction

                    # Generate prediction
                    prediction = classifier.predict(features)
                    logger.info(f"Prediction for {self.symbol}: {prediction[0]}")

                logger.info(f"{self.symbol} rolling window:\n{rolling_window_df.tail(3)}")

            # Clear interval data and update aggregation time
            self.interval_data = []
            self.next_aggregation_time += interval

    def save_to_csv(self):
        df = pd.DataFrame(self.aggregated_data)
        df.to_csv(f"{self.symbol}_5_minute_lookahead_data.csv", index=False)
        logger.info(f"Data for {self.symbol} saved to '{self.symbol}_5_minute_lookahead_data.csv'.")

# Create SymbolProcessor instances for each symbol
buffer_size = 3 # CHANGE TO 26 FOR FINAL IMPLEMENTATION
symbol_processors = {symbol: SymbolProcessor(symbol, buffer_size) for symbol in symbols}

# Async handler for incoming 1-minute bar data
async def bar_data_handler(bar):
    current_time = datetime.now()
    logger.info(f"Received bar data for symbol: {bar.symbol}")

    if bar.symbol in symbol_processors:
        await symbol_processors[bar.symbol].process_bar(bar, current_time)

# Subscribe to 1-minute bar data for each symbol
for symbol in symbols:
    wss_client.subscribe_bars(bar_data_handler, symbol)

# Function to run the WebSocket and stop on user input
async def run_stream():
    try:
        await asyncio.gather(
            wss_client._run_forever(),
            monitor_user_input()  # Function to stop streaming upon user input
        )
    except asyncio.CancelledError:
        logger.info("Streaming stopped by user.")
        # Save aggregated data to CSV files for each symbol
        for processor in symbol_processors.values():
            processor.save_to_csv()

# Helper function to monitor user input to stop the WebSocket
async def monitor_user_input():
    await asyncio.to_thread(input, "Press Enter to stop streaming...\n")
    await wss_client.stop()

# Main entry point to use the current event loop
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_stream())
