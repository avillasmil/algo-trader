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
import traceback
import threading
import time as tm

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
symbol_processors = {
    symbol: SymbolProcessor(
        symbol, model_name, wss_client, allocation_percentage, starting_cash, logger, buffer_size
    )
    for symbol in symbols
}

# Trading hours in Eastern Time
TRADING_START = time(9, 30)
TRADING_END = time(16, 0)
EASTERN_TZ = pytz.timezone("America/New_York")


def within_trading_hours():
    """Check if the current time is within trading hours (Eastern Time), excluding weekends."""
    now = datetime.now(EASTERN_TZ)
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    return TRADING_START <= now.time() <= TRADING_END


# Async handler for incoming 1-minute bar data
async def bar_data_handler(bar):
    current_time = datetime.now()
    logger.info(f"Received bar data for symbol: {bar.symbol}")

    if bar.symbol in symbol_processors:
        await symbol_processors[bar.symbol].process_bar(bar, current_time)


# Subscribe to 1-minute bar data for each symbol
for symbol in symbols:
    wss_client.subscribe_bars(bar_data_handler, symbol)


def run_stream():
    """
    Manage the WebSocket client and trading logic using threading for WebSocket handling.
    """
    def websocket_thread():
        """
        Run the WebSocket client in a separate thread.
        """
        try:
            logger.info("Starting WebSocket client.")
            wss_client.run()  # Blocking call to start the WebSocket
        except Exception as e:
            logger.error(f"Error in WebSocket thread: {e}\n{traceback.format_exc()}")
        finally:
            logger.info("WebSocket client has stopped.")

    try:
        while True:
            if within_trading_hours():
                logger.info("Market is open. Starting data stream and trading.")

                # Start the WebSocket thread
                thread = threading.Thread(target=websocket_thread, daemon=True)
                thread.start()

                # Wait until market close
                wait_until_market_close()

                # Stop the WebSocket client
                logger.info("Market closed. Stopping WebSocket client.")
                wss_client.stop()

                # Wait for the WebSocket thread to finish
                thread.join()

                logger.info("Market close processing complete.")
            else:
                logger.info("Market is closed. Waiting until next market open.")
                wait_until_market_open()
    except KeyboardInterrupt:
        logger.info("Streaming stopped by user.")
    except Exception as e:
        logger.error(f"Error in streaming loop: {e}\n{traceback.format_exc()}")


def wait_until_market_close():
    """
    Block until the market closes.
    """
    now = datetime.now(EASTERN_TZ)
    market_close = datetime.combine(now.date(), TRADING_END, tzinfo=EASTERN_TZ)
    time_to_close = (market_close - now).total_seconds()
    if time_to_close > 0:
        logger.info(f"Waiting for market close in {time_to_close // 60} minutes.")
        tm.sleep(time_to_close)


def wait_until_market_open():
    """
    Block until the market opens.
    """
    now = datetime.now(EASTERN_TZ)
    next_open = datetime.combine(now.date() + timedelta(days=1), TRADING_START, tzinfo=EASTERN_TZ)
    time_to_open = (next_open - now).total_seconds()
    logger.info(f"Waiting for market open in {time_to_open // 3600} hours and {(time_to_open % 3600) // 60} minutes.")
    tm.sleep(time_to_open)


if __name__ == "__main__":
    run_stream()
