"""
Data preprocessing module for Higgs boson dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def handle_missing_values(data, missing_value=-9999):
    """
    Replace missing value indicators with NaN and fill with median.
    
    Args:
        data: DataFrame to process
        missing_value: Value used to indicate missing data. Default: -9999
        
    Returns:
        DataFrame with missing values handled
    """
    data_clean = data.copy()
    data_clean = data_clean.replace(missing_value, np.nan)
    
    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data_clean[col].isnull().any():
            data_clean[col] = data_clean[col].fillna(data_clean[col].median())
    
    return data_clean


def split_features_labels(data, label_column='label'):
    """
    Split data into features and labels.
    
    Args:
        data: DataFrame with features and labels
        label_column: Name of the label column. Default: 'label'
        
    Returns:
        Tuple of (features, labels)
    """
    if label_column in data.columns:
        features = data.drop(columns=[label_column])
        labels = data[label_column]
    else:
        features = data
        labels = None
    
    return features, labels


def preprocess_data(data, label_column='label', missing_value=-9999):
    """
    Complete preprocessing pipeline.
    
    Args:
        data: DataFrame to preprocess
        label_column: Name of the label column. Default: 'label'
        missing_value: Value used to indicate missing data. Default: -9999
        
    Returns:
        Tuple of (features, labels)
    """
    data_clean = handle_missing_values(data, missing_value)
    features, labels = split_features_labels(data_clean, label_column)
    return features, labels


def combine_datasets(data_dict):
    """
    Combine multiple datasets from dictionary into single DataFrame.
    
    Args:
        data_dict: Nested dictionary with structure data[channel][process]
        
    Returns:
        Combined DataFrame with all data
    """
    all_data = []
    
    for channel in data_dict:
        for process in data_dict[channel]:
            df = data_dict[channel][process].copy()
            df['channel'] = channel
            df['process'] = process
            all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def scale_features(x_train, x_test=None):
    """
    Standardise features using StandardScaler.

    Args:
        x_train: Training features
        x_test: Test features (optional)
    
    Returns:
        Scaled training features and optionally scaled test features
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    
    if x_test is not None:
        x_test_scaled = scaler.transform(x_test)
        return x_train_scaled, x_test_scaled, scaler
    
    return x_train_scaled, scaler


def prepare_train_test_split(data, test_size=0.2, random_state=42, label_column='label'):
    """
    Split data into train and test sets.
    
    Args:
        data: DataFrame with features and labels
        test_size: Proportion of data for test set. Default: 0.2
        random_state: Random seed. Default: 42
        label_column: Name of label column. Default: 'label'
        
    Returns:
        x_train, x_test, y_train, y_test
    """
    features, labels = split_features_labels(data, label_column)
    
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if labels is not None else None
    )
    
    return x_train, x_test, y_train, y_test


def scale_histograms_to_correct_yield(histograms):
    """
    Scale histograms for different processes to the expected yields.
    
    Args:
        histograms (dict): Dictionary of histograms with keys "Z", "ggH", "VBF".
                           Each value is a NumPy array representing bin counts.

    Returns:
        dict: The input dictionary with histograms scaled to the expected yields.
    """
    for proc in ["Z", "ggH", "VBF"]:
        histograms[proc] = histograms[proc].astype(float)
    histograms["Z"] *= 8.4
    histograms["ggH"] *= 0.034
    histograms["VBF"] *= 0.011
    return histograms


def check_events_numbers(histograms, threshold=10):
    """
    Check that each bin in the summed histograms has at least a minimum number of events.

    Args:
        histograms (dict): Dictionary of histograms with process names as keys and
                           NumPy arrays of bin counts as values.
        threshold (int, optional): Minimum number of events required in each bin. Default is 10.

    Raises:
        ValueError: If any bin in the summed histogram has fewer than `threshold` events.
    """
    summed_histogram = np.sum(list(histograms.values()), axis=0)
    if (summed_histogram < threshold).any():
        raise ValueError("Not enough events in one or more of the histogram bins")
