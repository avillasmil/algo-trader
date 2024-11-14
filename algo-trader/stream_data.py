import asyncio
import pickle
import logging
from datetime import datetime, time, timedelta
import pytz
from alpaca.data.live import StockDataStream
import pandas as pd
from collections import defaultdict, deque
from data_fetch import get_paper_creds
from SymbolProcessor import SymbolProcessor


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

# Trading symbols and parameters
symbols = ["MSFT", "AAPL", "GOOG"]
model_name = "model_xgb_e67d3402"
allocation_percentage = 1 / len(symbols)
starting_cash = 10000
buffer_size = 26

# Initialize WebSocket client and SymbolProcessors
wss_client = StockDataStream(API_KEY, SECRET_KEY)
symbol_processors = {symbol: SymbolProcessor(symbol, model_name, wss_client, allocation_percentage, starting_cash, logger, buffer_size) for symbol in symbols}

# Trading hours in Eastern Time
TRADING_START = time(9, 30)
TRADING_END = time(16, 0)
EASTERN_TZ = pytz.timezone("America/New_York")

def within_trading_hours():
    """Check if the current time is within trading hours (Eastern Time)."""
    current_time = datetime.now(EASTERN_TZ).time()
    return TRADING_START <= current_time <= TRADING_END

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
        while True:
            if within_trading_hours():
                logger.info("Market is open. Starting data stream and trading.")
                await wss_client.run()  # Start the WebSocket connection
                await wait_until_market_close()     # Wait until market close
                await wss_client.stop()             # Stop the WebSocket at market close
                logger.info("Market closed. Stopping data stream.")
            else:
                time_to_open = time_until_market_open()
                logger.info(f"Market is closed. Waiting {time_to_open // 3600} hours and {(time_to_open % 3600) // 60} minutes until next open.")
                await asyncio.sleep(time_to_open)  # Sleep until the next market open
    except asyncio.CancelledError:
        logger.info("Streaming stopped by user.")
        # Save aggregated data to CSV files for each symbol
        for processor in symbol_processors.values():
            processor.save_to_csv()
    except Exception as e:
        logger.error(f"Error in streaming loop: {e}")

def time_until_market_open():
    """Calculate the seconds until the next market open."""
    now = datetime.now(EASTERN_TZ)
    if now.time() >= TRADING_END:  # Market already closed today, wait until next day
        next_open = datetime.combine(now.date() + timedelta(days=1), TRADING_START, tzinfo=EASTERN_TZ)
    elif now.time() < TRADING_START:  # Before market open today
        next_open = datetime.combine(now.date(), TRADING_START, tzinfo=EASTERN_TZ)
    else:
        return 0  # Market is open; no waiting needed
    return (next_open - now).total_seconds()


# Helper function to monitor user input to stop the WebSocket
async def monitor_user_input():
    await asyncio.to_thread(input, "Press Enter to stop streaming...\n")
    await wss_client.stop()

# Main entry point to use the current event loop
if __name__ == "__main__":
    asyncio.run(run_stream())
