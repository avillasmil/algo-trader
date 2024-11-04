from pandas import DataFrame

def train_test_split(df: DataFrame, validation_split: float):
    if validation_split > 0:
        train_cutoff = int(len(df) * (1 - validation_split))
        train_data = df.iloc[:train_cutoff]
        val_data = df.iloc[train_cutoff:]
    else:
        train_data = df
        val_data = None
    return train_data, val_data