import asyncio
import pickle
import logging
from alpaca.data.live import StockDataStream
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict, deque
from data_fetch import get_paper_creds
from SymbolProcessor import SymbolProcessor
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

symbols = ["MSFT", "AAPL", "GOOG"] # Symbols to subscribe to
model_name = "model_xgb_e67d3402" # Trading model to use
allocation_percentage = 1/len(symbols) # Allocation percentage per symbol(uniform)
starting_cash = 10000

# Initialize WebSocket client
wss_client = StockDataStream(API_KEY, SECRET_KEY)

# Create SymbolProcessor instances for each symbol
buffer_size = 26 # CHANGE TO 26 FOR FINAL IMPLEMENTATION
symbol_processors = {symbol: SymbolProcessor(symbol, model_name, wss_client, allocation_percentage, starting_cash, logger, buffer_size) for symbol in symbols}

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
