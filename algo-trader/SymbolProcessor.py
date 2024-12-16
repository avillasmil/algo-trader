import asyncio
import pickle
import logging
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from preprocess import (
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_obv,
    calculate_macd
)

class SymbolProcessor:
    def __init__(self, symbol, model_name, alpaca_client, allocation_percentage, starting_cash, logger, buffer_size=26):
        self.symbol = symbol
        self.interval = timedelta(minutes=5)
        self.logger = logger
        self.buffer_size = buffer_size
        self.next_aggregation_time = None
        self.interval_data = []
        self.rolling_window = deque(maxlen=buffer_size)
        self.obv_value = 0
        self.alpaca_client = alpaca_client

        # Initialize cash and position allocation
        self.allocation_percentage = allocation_percentage
        self.starting_cash = starting_cash
        self.cash_available = self.starting_cash * self.allocation_percentage
        self.current_position = 0.0

        # Load trained classifier model
        with open(f"models/{model_name}.pkl", "rb") as f:
            self.classifier = pickle.load(f)

        # Initialize holdings based on current Alpaca account
        self.initialize_holdings()

    def initialize_holdings(self):
        """Check current holdings for the symbol and initialize position and cash accordingly."""
        try:
            # Get account details and check if there are holdings for this symbol
            position = self.alpaca_client.get_position(self.symbol)
            if position:
                self.current_position = float(position.qty)
                self.cash_available = 0.0
                self.logger.info(f"Initialized position for {self.symbol}: {self.current_position} shares held.")
            else:
                self.logger.info(f"No current holdings for {self.symbol}. Cash available: ${self.cash_available}")
        except Exception as e:
            # Interpret exception as no holdings if symbol not found in account
            self.logger.info(f"No holdings found for {self.symbol}. Exception: {e}")
            self.current_position = 0.0

    def initialize_aggregation_time(self, current_time):
        self.next_aggregation_time = current_time.replace(second=0, microsecond=0)
        if self.next_aggregation_time.minute % 5 != 0:
            self.next_aggregation_time += timedelta(minutes=(5 - self.next_aggregation_time.minute % 5))

    async def process_bar(self, bar, current_time):
        self.interval_data.append(bar)

        if self.next_aggregation_time is None:
            self.initialize_aggregation_time(current_time)

        if current_time >= self.next_aggregation_time + self.interval:
            if self.interval_data:
                close_price = self.interval_data[-1].close
                total_volume = sum(item.volume for item in self.interval_data)

                new_row = {"timestamp": self.next_aggregation_time, "close": close_price, "volume": total_volume}
                
                if len(self.rolling_window) > 0:
                    last_close = self.rolling_window[-1]["close"]
                    if close_price > last_close:
                        self.obv_value += total_volume
                    elif close_price < last_close:
                        self.obv_value -= total_volume

                new_row["obv"] = self.obv_value
                self.rolling_window.append(new_row)

                rolling_window_df = pd.DataFrame(list(self.rolling_window))
                if len(self.rolling_window) == self.buffer_size:
                    rolling_window_df["bb"] = calculate_bollinger_bands(rolling_window_df["close"])
                    rolling_window_df["rsi"] = calculate_rsi(rolling_window_df["close"])
                    rolling_window_df["sma"] = calculate_sma(rolling_window_df["close"])
                    rolling_window_df["ema"] = calculate_ema(rolling_window_df["close"])
                    macd_df = calculate_macd(rolling_window_df["close"])
                    rolling_window_df = pd.concat([rolling_window_df, macd_df], axis=1)

                    feature_row = rolling_window_df.iloc[-1][["close", "volume", "bb", "rsi", "sma", "ema", "obv", "MACD", "Signal", "Histogram"]]
                    features = feature_row.values.reshape(1, -1)

                    prediction = self.classifier.predict(features)
                    self.logger.info(f"Prediction for {self.symbol}: {prediction[0]}")
                    
                    await self.execute_trade(prediction[0], close_price)

                self.logger.info(f"{self.symbol} rolling window:\n{rolling_window_df.tail(3)}")

            self.interval_data = []
            self.next_aggregation_time += self.interval

    async def execute_trade(self, prediction, close_price):
        if prediction == 1:
            if self.cash_available > 0:
                quantity_to_buy = self.cash_available / close_price
                if quantity_to_buy > 0:
                    await self.place_order(quantity_to_buy, OrderSide.BUY)
                    self.cash_available -= quantity_to_buy * close_price
                    self.current_position += quantity_to_buy
                    self.logger.info(f"Bought {quantity_to_buy} shares of {self.symbol} at ${close_price}")

        elif prediction == 2:
            if self.current_position > 0:
                await self.place_order(self.current_position, OrderSide.SELL)
                self.cash_available += self.current_position * close_price
                self.logger.info(f"Sold {self.current_position} shares of {self.symbol} at ${close_price}")
                self.current_position = 0.0

    async def place_order(self, quantity, side):
        order = MarketOrderRequest(
            symbol=self.symbol,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        self.alpaca_client.submit_order(order)

