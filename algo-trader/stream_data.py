import asyncio
from alpaca.data.live import StockDataStream
from datetime import datetime, timedelta
import pandas as pd

# Alpaca API credentials
API_KEY = ''
SECRET_KEY = ''

# Initialize WebSocket client
wss_client = StockDataStream(API_KEY, SECRET_KEY)

# Variables for 5-minute look-ahead aggregation
interval = timedelta(minutes=5)
next_aggregation_time = None
interval_data = []

# DataFrame to store 5-minute bar data
columns = ["Timestamp", "Close Price", "Total Volume"]
aggregated_df = pd.DataFrame(columns=columns)

# Async handler for incoming 1-minute bar data
async def bar_data_handler(data):
    global next_aggregation_time, interval_data, aggregated_df
    current_time = datetime.now()

    # Initialize next_aggregation_time to the next "5-minute mark"
    if next_aggregation_time is None:
        next_aggregation_time = current_time.replace(second=0, microsecond=0)
        if next_aggregation_time.minute % 5 != 0:
            next_aggregation_time += timedelta(minutes=(5 - next_aggregation_time.minute % 5))

    # Collect data within the 5-minute interval
    interval_data.append(data)

    # Check if current time has reached the end of the 5-minute interval
    if current_time >= next_aggregation_time + interval:
        # Aggregate data for close price and volume
        if interval_data:
            close_price = interval_data[-1].close  # Last price within the 5-minute interval
            total_volume = sum(item.volume for item in interval_data)
            
            # Add data to DataFrame with the start time of the 5-minute interval
            new_row = {"Timestamp": next_aggregation_time, "Close Price": close_price, "Total Volume": total_volume}
            aggregated_df = pd.concat([aggregated_df, pd.DataFrame([new_row])], ignore_index=True)
            
            print(f"5-minute aggregated data at {next_aggregation_time}:")
            print(f"Close Price: {close_price}, Total Volume: {total_volume}")

            # Clear interval data and set the next aggregation time
            interval_data = []
            next_aggregation_time += interval

# Subscribe to 1-minute bar data for MSFT
wss_client.subscribe_bars(bar_data_handler, "MSFT")

# Function to run the WebSocket and stop on user input
async def run_stream():
    try:
        await asyncio.gather(
            wss_client._run_forever(),
            monitor_user_input()  # Function to stop streaming upon user input
        )
    except asyncio.CancelledError:
        print("Streaming stopped by user.")
        # Save the DataFrame to a CSV file when the stream stops
        aggregated_df.to_csv("MSFT_5_minute_lookahead_data.csv", index=False)
        print("Data saved to 'MSFT_5_minute_lookahead_data.csv'.")

# Helper function to monitor user input to stop the WebSocket
async def monitor_user_input():
    await asyncio.to_thread(input, "Press Enter to stop streaming...\n")
    await wss_client.stop()

# Main entry point to use the current event loop
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_stream())
