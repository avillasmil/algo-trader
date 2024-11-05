import json
import hashlib

def split_features_and_labels(df, label_column='label'):
    """
    Splits a dataframe into features and labels.
    """
    X = df.drop(columns=[label_column]).reset_index(drop=True)
    y = df[label_column].reset_index(drop=True)
    return X, y

def get_model_filename(config: dict, base_path="models/") -> str:
    # Convert config dictionary to a sorted JSON string to ensure consistency
    config_str = json.dumps(config, sort_keys=True)
    
    # Generate a hash of the configuration
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    # Create filename with a readable part and the hash
    filename = f"{base_path}model_{config['model_to_use']}_{config_hash[:8]}.pkl"
    return filename

def get_sample_weights(y_train):
    class_weights = y_train.value_counts(normalize=True)  # Class distribution
    total_samples = len(y_train)
    scale_pos_weight = total_samples / (len(class_weights) * class_weights)
    sample_weight = y_train.map(scale_pos_weight)
    return sample_weight