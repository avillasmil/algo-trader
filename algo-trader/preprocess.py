import pandas as pd

def train_test_split(df: pd.DataFrame, validation_split: float):
    if validation_split > 0:
        train_cutoff = int(len(df) * (1 - validation_split))
        train_data = df.iloc[:train_cutoff]
        val_data = df.iloc[train_cutoff:]
    else:
        train_data = df
        val_data = None
    return train_data, val_data

def calculate_bollinger_bands(data, window=24, num_of_std=2):
    """Calculate Bollinger Bands ratio wrt current price"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    bb_ratio = (data - rolling_mean) / (rolling_std * num_of_std)
    return bb_ratio

def calculate_rsi(data, window=24):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data, window=24):
    """Calculate SMA ratio of current price."""
    rolling_mean = data.rolling(window=window).mean()
    sma = (data / rolling_mean) - 1
    return sma

def calculate_obv(data):
    # Initialize OBV series with the same index as the DataFrame
    obv = [0]
    
    # Loop through each row in the DataFrame
    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['close'].iloc[i - 1]:
            # Price went up, add the volume
            obv.append(obv[-1] + data['volume'].iloc[i])
        elif data['close'].iloc[i] < data['close'].iloc[i - 1]:
            # Price went down, subtract the volume
            obv.append(obv[-1] - data['volume'].iloc[i])
        else:
            # Price stayed the same, OBV remains unchanged
            obv.append(obv[-1]) 
    return obv

def calculate_ema(data, window = 24):
    """Calculate EMA ratio of current price."""
    rolling_mean = data.ewm(span=window, adjust=False).mean()
    ema = (data / rolling_mean) - 1
    return ema

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD line, Signal line, and MACD Histogram.
    """
    # Calculate the short and long EMAs
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    
    # Calculate the MACD line
    macd_line = short_ema - long_ema
    
    # Calculate the Signal line
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    # Calculate the MACD Histogram
    macd_histogram = macd_line - signal_line
    
    # Combine the results in a DataFrame
    macd_df = pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': macd_histogram
    })
    
    return macd_df

def bollinger_detection(bb, thresh = 1):
    """
    Assign labels based on bollinger band crossings.
    """
    # Initialize the result list with zeros
    crossings = [0] * len(bb)
    
    # Loop through each element, starting from the second element (index 1)
    for i in range(1, len(bb)):
        # Check for -1 crossing: previous value <= -1 and current value > -1
        if bb.iloc[i-1] <= -thresh and bb.iloc[i] > -thresh:
            crossings[i] = 1
        # Check for 1 crossing: previous value >= 1 and current value < 1
        elif bb.iloc[i-1] >= thresh and bb.iloc[i] < thresh:
            crossings[i] = 2
    
    return pd.Series(crossings, index=bb.index)

def rsi_detection(rsi, low_thresh = 30, high_thresh = 70):
    """
    Assign labels based on RSI crossings.
    """
    # Initialize the result list with zeros
    crossings = [0] * len(rsi)
    
    # Loop through each element, starting from the second element (index 1)
    for i in range(1, len(rsi)):
        # Check for low crossing: previous value <= low_thresh and current value > low_thresh
        if rsi.iloc[i-1] <= low_thresh and rsi.iloc[i] > low_thresh:
            crossings[i] = 1
        # Check for high crossing: previous value >= 1 and current value < 1
        elif rsi.iloc[i-1] >= high_thresh and rsi.iloc[i] < high_thresh:
            crossings[i] = 2
    
    return pd.Series(crossings, index=rsi.index)

def generate_labels(df):
    """
    Generate trading labels for algorithm. 0 = hold, 1 = buy, 2 = sell.
    """
    labels_bb = bollinger_detection(df["bb"])
    labels_rsi = rsi_detection(df["rsi"])
    
    labels_final = pd.Series(
        [val if val == labels_bb.iloc[i] else 0 for i, val in enumerate(labels_rsi)],
        index=labels_rsi.index
    )
    return labels_final

def generate_label_summary(train_data, val_data):
    n_buy = (train_data['label'] == 1).sum()
    n_sell = (train_data['label'] == 2).sum()
    print(f"{n_buy} total train buy labels ({n_buy/len(train_data):.6f}%)") 
    print(f"{n_sell} total train sell labels ({n_sell/len(train_data):.6f}%)")

    n_buy = (val_data['label'] == 1).sum()
    n_sell = (val_data['label'] == 2).sum()
    print(f"{n_buy} total val buy labels ({n_buy/len(train_data):.6f}%)") 
    print(f"{n_sell} total val sell labels ({n_sell/len(train_data):.6f}%)")

    
    