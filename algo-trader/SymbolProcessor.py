import asyncio
import pickle
import logging
import pandas as pd
from collections import deque

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