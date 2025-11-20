"""
Data loader module for Higgs boson dataset.
"""

import pandas as pd


def load_data(channels=None, processes=None):
    """
    Load Higgs boson datasets from GitHub repository.
    
    Args:
        channels: List of decay channels to load. Default: ["et", "mt", "tt"]
        processes: List of processes to load. Default: ["Z", "ggH", "VBF"]
        
    Returns:
        Dictionary with structure data[channel][process] = DataFrame
    """
    if channels is None:
        channels = ["et", "mt", "tt"]
    if processes is None:
        processes = ["Z", "ggH", "VBF"]
    
    data = {}
    
    for channel in channels:
        data[channel] = {}
        for process in processes:
            url = f"https://raw.githubusercontent.com/gputtley/ML-Assessment-3/master/{channel}_{process}.pkl"
            data[channel][process] = pd.read_pickle(url)
    
    return data


def load_single_dataset(channel, process):
    """
    Load a single dataset for a specific channel and process.
    
    Args:
        channel: Decay channel (e.g., "et", "mt", "tt")
        process: Process type (e.g., "Z", "ggH", "VBF")
        
    Returns:
        DataFrame with the dataset
    """
    url = f"https://raw.githubusercontent.com/gputtley/ML-Assessment-3/master/{channel}_{process}.pkl"
    return pd.read_pickle(url)
