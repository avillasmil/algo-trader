import json
from preprocess import train_test_split
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient

def get_paper_creds():
    with open("/Users/alejandrovillasmil/Documents/GitHub/algo-trader/credentials.json", 'r') as file:
        creds = json.load(file)
    key = creds["ALPACA_PAPER_KEY"]
    secret = creds["ALPACA_PAPER_SECRET"]
    return key, secret

def fetch_symbol_data(symbol, period, train_start_end, validation_split, stock_historical_data_client):
    """
    Fetches and processes data for a single stock symbol (train & validation).
    """
    print(f"Fetching {symbol} data...")

    # Create request for stock bars with specified symbol, timeframe, and date range
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(amount=period, unit=TimeFrameUnit.Minute),
        start=train_start_end[0],
        end=train_start_end[1]
    )
    
    # Retrieve data and select only 'close' and 'volume' columns
    df = stock_historical_data_client.get_stock_bars(req).df.loc[:, ["close", "volume"]]
    
    # Split data into training and validation sets
    train_data, val_data = train_test_split(df, validation_split)
    return symbol, train_data, val_data


def fetch_symbol_data_BT(symbol, period, backtest_start_end, stock_historical_data_client):
    """
    Fetches and processes data for a single stock symbol (Backtesting).
    """
    print(f"Fetching {symbol} data...")

    # Create request for stock bars with specified symbol, timeframe, and date range
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(amount=period, unit=TimeFrameUnit.Minute),
        start=backtest_start_end[0],
        end=backtest_start_end[1]
    )
    
    # Retrieve data and select only 'close' and 'volume' columns
    df = stock_historical_data_client.get_stock_bars(req).df.loc[:, ["close", "volume"]]
    
    return symbol, df

