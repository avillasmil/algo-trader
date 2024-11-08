import asyncio
import pickle 
from alpaca.data.live import StockDataStream
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict, deque
import threading
from data_fetch import get_paper_creds
from preprocess import (
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_obv,
    calculate_macd,
)

# Alpaca API credentials
API_KEY, SECRET_KEY= get_paper_creds()
BASE_URL = 'https://paper-api.alpaca.markets'  # Use the paper trading URL or live URL

# Symbols to subscribe to
symbols = ["MSFT", "AAPL", "GOOG"] #AAPL, GOOG

# Initialize WebSocket client
wss_client = StockDataStream(API_KEY, SECRET_KEY)

# Variables for 5-minute look-ahead aggregation
interval = timedelta(minutes=5)
next_aggregation_time = defaultdict(lambda: None)  # Dictionary to track aggregation time per symbol
# next_aggregation_time = None
interval_data = defaultdict(list)  # To store data per symbol
rolling_window = defaultdict(lambda: deque(maxlen=26))  # Rolling window for each symbol

# DataFrame to store aggregated data
aggregated_df = defaultdict(pd.DataFrame)  # Dictionary of DataFrames for each symbol

# Load the trained ML model (assuming it's a classifier)
# with open("models/model_xgb_e67d3402.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# Async handler for incoming 1-minute bar data
async def bar_data_handler(bar):
    global next_aggregation_time, interval_data, aggregated_df, rolling_window
    current_time = datetime.now()
    print(f"Received bar data for symbol: {bar.symbol}")

    # Initialize next_aggregation_time for each symbol
    if next_aggregation_time[bar.symbol] is None:
        next_aggregation_time[bar.symbol] = current_time.replace(second=0, microsecond=0)
        if next_aggregation_time[bar.symbol].minute % 5 != 0:
            next_aggregation_time[bar.symbol] += timedelta(minutes=(5 - next_aggregation_time[bar.symbol].minute % 5))

    # Initialize next_aggregation_time to the next "5-minute mark"
    # if next_aggregation_time is None:
    #     next_aggregation_time = current_time.replace(second=0, microsecond=0)
    #     if next_aggregation_time.minute % 5 != 0:
    #         next_aggregation_time += timedelta(minutes=(5 - next_aggregation_time.minute % 5))

    # Process each symbol concurrently
    tasks = []
    for symbol in symbols:
        if bar.symbol == symbol:
            tasks.append(process_symbol(symbol, bar, current_time))

    # Run all the symbol processing tasks concurrently
    await asyncio.gather(*tasks)

# Function to process a symbol's data (aggregation, prediction)
async def process_symbol(symbol, bar, current_time):
    global next_aggregation_time, interval_data, aggregated_df, rolling_window

    interval_data[symbol].append(bar)

    # Check if current time has reached the end of the 5-minute interval for each symbol
    if current_time >= next_aggregation_time[symbol] + interval:
        if interval_data[symbol]:
            # Aggregate data for close price and volume
            close_price = interval_data[symbol][-1].close  # Last price within the 5-minute interval
            total_volume = sum(item.volume for item in interval_data[symbol])

            # Create the new row with the aggregated data
            new_row = {"timestamp": next_aggregation_time[symbol], "close": close_price, "volume": total_volume}

            # Append the new row to both DataFrame and rolling window for each symbol
            aggregated_df[symbol] = pd.concat([aggregated_df[symbol], pd.DataFrame([new_row])], ignore_index=True)
            rolling_window[symbol].append(new_row)

            # Convert deque to DataFrame for each symbol
            rolling_window_df = pd.DataFrame(list(rolling_window[symbol]))

            # Calculate additional features using the technical indicator functions
            if len(rolling_window[symbol]) > 1:
                rolling_window_df["Bollinger Bands"] = calculate_bollinger_bands(rolling_window_df["close"], window =2)
                rolling_window_df["RSI"] = calculate_rsi(rolling_window_df["close"], window =2)
                rolling_window_df["SMA"] = calculate_sma(rolling_window_df["close"], window =2)
                rolling_window_df["EMA"] = calculate_ema(rolling_window_df["close"], window =2)
                rolling_window_df["OBV"] = calculate_obv(rolling_window_df)

                # Calculate MACD and concatenate as new columns to rolling_window_df
                # macd_df = calculate_macd(close_prices)
                # rolling_window_df = pd.concat([rolling_window_df, macd_df], axis=1)  # Concatenate MACD as new columns

            print(f"Rolling window for {symbol} contains {len(rolling_window_df)} samples with features:")
            print(rolling_window_df.tail(3))  # Show the latest 3 rows for debugging

            # Make prediction on the most recent row (latest 5-minute data)
            # recent_data = rolling_window_df.iloc[-1:].drop(columns=["Timestamp"])  # Drop the Timestamp column
            # prediction = model.predict(recent_data)[0]  # Predict buy/sell/hold

            # print(f"Prediction for {symbol}: {prediction} (Buy=1, Sell=2, Hold=0)")

        # Clear interval data and set the next aggregation time for each symbol
        interval_data[symbol] = []
        next_aggregation_time[symbol] += interval

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
        print("Streaming stopped by user.")
        # Save the DataFrame to CSV files for each symbol when the stream stops
        for symbol in symbols:
            aggregated_df[symbol].to_csv(f"{symbol}_5_minute_lookahead_data.csv", index=False)
            print(f"Data for {symbol} saved to '{symbol}_5_minute_lookahead_data.csv'.")

# Helper function to monitor user input to stop the WebSocket
async def monitor_user_input():
    await asyncio.to_thread(input, "Press Enter to stop streaming...\n")
    await wss_client.stop()

# Main entry point to use the current event loop
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_stream())
