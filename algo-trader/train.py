def split_features_and_labels(df, label_column='label'):
    """
    Splits a dataframe into features and labels.
    """
    X = df.drop(columns=[label_column]).reset_index(drop=True)
    y = df[label_column].reset_index(drop=True)
    return X, y
