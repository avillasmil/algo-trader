import asyncio
import pickle
import logging
import pandas as pd
from collections import deque
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class SymbolProcessor:
    def __init__(self, symbol, model_name, alpaca_client, allocation_percentage, starting_cash, buffer_size=26):
        self.symbol = symbol
        self.buffer_size = buffer_size
        self.next_aggregation_time = None
        self.interval_data = []
        self.rolling_window = deque(maxlen=buffer_size)
        self.obv_value = 0
        self.alpaca_client = alpaca_client  # Alpaca trading client

        # Initial funds allocation
        self.allocation_percentage = allocation_percentage
        self.cash_available = starting_cash * allocation_percentage  # Initial cash allocated to this symbol
        self.current_position = 0.0  # Fractional shares held

        # Load trained classifier model
        with open(f"models/{model_name}.pkl", "rb") as f:
            self.classifier = pickle.load(f)

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

                rolling_window_df = pd.DataFrame(list(self.rolling_window))
                # Calculate technical indicators
                if len(self.rolling_window) == self.buffer_size:
                    rolling_window_df["bb"] = calculate_bollinger_bands(rolling_window_df["close"], window=2)
                    rolling_window_df["rsi"] = calculate_rsi(rolling_window_df["close"], window=2)
                    rolling_window_df["sma"] = calculate_sma(rolling_window_df["close"], window=2)
                    rolling_window_df["ema"] = calculate_ema(rolling_window_df["close"], window=2)
                    macd_df = calculate_macd(rolling_window_df["close"], short_window=1, long_window=3, signal_window=2)
                    rolling_window_df = pd.concat([rolling_window_df, macd_df], axis=1)

                    # Extract the last row as feature input for classifier
                    feature_row = rolling_window_df.iloc[-1][["close", "volume", "bb", "rsi", "sma", "ema", "obv", "MACD", "Signal", "Histogram"]]
                    features = feature_row.values.reshape(1, -1)  # Reshape for a single prediction

                    # Generate prediction
                    prediction = self.classifier.predict(features)
                    logger.info(f"Prediction for {self.symbol}: {prediction[0]}")
                    
                    # Execute trade based on prediction
                    await self.execute_trade(prediction[0], close_price)

                logger.info(f"{self.symbol} rolling window:\n{rolling_window_df.tail(3)}")

            # Clear interval data and update aggregation time
            self.interval_data = []
            self.next_aggregation_time += interval

    async def execute_trade(self, prediction, close_price):
        """Execute buy or sell based on the classifier's prediction."""
        if prediction == 1:  # Buy action
            if self.cash_available > 0:  # Check if funds are sufficient
                # Calculate the fraction of a share to buy with the available cash
                quantity_to_buy = self.cash_available / close_price
                if quantity_to_buy > 0:
                    await self.place_order(quantity_to_buy, OrderSide.BUY)
                    self.cash_available -= quantity_to_buy * close_price
                    self.current_position += quantity_to_buy
                    logger.info(f"Bought {quantity_to_buy} shares of {self.symbol} at ${close_price}")

        elif prediction == 2:  # Sell action
            if self.current_position > 0:
                await self.place_order(self.current_position, OrderSide.SELL)
                self.cash_available += self.current_position * close_price
                logger.info(f"Sold {self.current_position} shares of {self.symbol} at ${close_price}")
                self.current_position = 0.0  # Reset position to zero

    async def place_order(self, quantity, side):
        """Place an order through Alpaca, allowing fractional quantities."""
        order = MarketOrderRequest(
            symbol=self.symbol,
            qty=quantity,  # Allows fractional quantities
            side=side,
            time_in_force=TimeInForce.DAY
        )
        self.alpaca_client.submit_order(order)
