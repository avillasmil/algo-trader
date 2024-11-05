from preprocess import (
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_obv,
    calculate_macd,
)
import pandas as pd

def generate_BT_features(data, symbols):
    """
    Generates backtesting data with technical indicators for a list of stock symbols.
    
    Parameters:
    - data (dict): Dictionary where each key is a stock symbol, and the value is a DataFrame 
                   containing stock data (e.g., 'close' prices).
    - symbols (list): List of stock symbols to process.
    
    Returns:
    - dict: Updated dictionary with each symbol's DataFrame including calculated indicators.
    """
    # Indicator functions and column names to add
    indicator_funcs = [
        ('bb', calculate_bollinger_bands),
        ('sma', calculate_sma),
        ('rsi', calculate_rsi),
        ('obv', calculate_obv),
        ('ema', calculate_ema),
    ]
    
    for symbol in symbols:
        df = data[symbol]
        
        # Apply each indicator function and assign the result as a new column
        df = df.assign(
            **{name: func(df['close']) if name != 'obv' else func(df) for name, func in indicator_funcs}
        )
        
        # Add MACD separately and concatenate with the existing DataFrame
        macd_df = calculate_macd(df['close'])
        df = pd.concat([df, macd_df], axis=1)
        
        # Drop any NaN values created in indicator calculation
        df.dropna(inplace=True)
        
        # Update the symbol's data with the processed DataFrame
        data[symbol] = df
        
    return data

def preprocess_BT_data(historical_data):
    """ Reset df index & remove duplicate rows. """
    for symbol, df in historical_data.items():
        # Reset the index to turn multi-index into columns
        df = df.reset_index()
        
        # Set 'timestamp' as the single index
        df = df.set_index('timestamp')
        df.drop(columns = 'symbol',inplace=True)

        if df.index.duplicated().any():
            # Remove duplicates, keeping only the first occurrence of each timestamp
            df = df[~df.index.duplicated(keep='first')]
        
        # Update the dictionary with the modified dataframe
        historical_data[symbol] = df
    return historical_data


def run_simulation(historical_data, symbols, model, portfolio, initial_cash):
    """
    Runs a stock trading simulation based on a model's predictions and updates the portfolio accordingly.

    Parameters:
    - historical_data (dict): Dictionary of DataFrames where each key is a stock symbol, 
                              and the value is the historical stock data with dates as index.
    - symbols (list): List of stock symbols to trade.
    - model: Trained model that provides buy (1), sell (2), or hold (0) predictions.
    - portfolio (dict): Portfolio dictionary with 'cash', 'positions', and 'history' to track investments.
    - initial_cash (float): Initial cash amount to calculate portfolio return.

    Returns:
    - dict: Updated portfolio with transaction history and final values.
    """
    for date in historical_data[symbols[0]].index:  # Assumes all stocks have the same dates
        portfolio_value = sum(portfolio['cash'].values())

        for symbol in symbols:
            stock_data = historical_data[symbol]

            if date in stock_data.index:
                # Fetch current price and prepare row data for prediction
                current_price = stock_data.at[date, 'close']
                row_data = stock_data.loc[date].to_frame().T.reset_index(drop=True)
                action = model.predict(row_data)

                # Buy logic
                if action == 1:
                    max_shares_to_buy = portfolio['cash'][symbol] / current_price
                    if max_shares_to_buy > 0:
                        portfolio['positions'][symbol]['shares'] += max_shares_to_buy
                        portfolio['cash'][symbol] -= max_shares_to_buy * current_price
                        print(f"Buying {max_shares_to_buy:.4f} shares of {symbol} at ${current_price:.2f} on {date}")

                # Sell logic
                elif action == 2:
                    shares_held = portfolio['positions'][symbol]['shares']
                    if shares_held > 0:
                        portfolio['positions'][symbol]['shares'] = 0
                        portfolio['cash'][symbol] += shares_held * current_price
                        print(f"Selling {shares_held:.4f} shares of {symbol} at ${current_price:.2f} on {date}")

                # Update position value in portfolio
                shares = portfolio['positions'][symbol]['shares']
                portfolio['positions'][symbol]['value'] = shares * current_price
                portfolio_value += portfolio['positions'][symbol]['value']

        # Record daily portfolio value
        portfolio['history'].append({'date': date, 'portfolio_value': portfolio_value})

    # Calculate the final portfolio return
    final_value = portfolio['history'][-1]['portfolio_value']
    portfolio_return = (final_value - initial_cash) / initial_cash * 100

    # Print summary
    print(f"\nInitial Portfolio Value: ${initial_cash:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Portfolio Return: {portfolio_return:.2f}%")

    # Return updated portfolio with final return value
    portfolio['final_value'] = final_value
    portfolio['return'] = portfolio_return
    return portfolio


def calculate_baseline_return(symbols, cash_allocation, historical_data):
    """
    Calculate the baseline return based on the provided symbols, cash allocation,
    and historical data. Hold only return.

    Args:
        symbols (list): A list of stock symbols to consider.
        cash_allocation (dict): A dictionary mapping each symbol to its cash allocation.
        historical_data (dict): A dictionary containing historical price data for each symbol.
        
    """
    return_BL = 0  # Initialize baseline return

    # Iterate over each symbol to calculate its contribution to the baseline return
    for symbol in symbols:
        alloc = cash_allocation[symbol]  # Get the cash allocation for the symbol
        # Calculate the baseline return for the symbol
        bl = ((historical_data[symbol]["close"].iloc[-1] - historical_data[symbol]["close"].iloc[0]) /
              historical_data[symbol]["close"].iloc[0]) * 100
        
        # Accumulate the weighted baseline return
        return_BL += alloc * bl

    return return_BL  # Return the total baseline return

    
