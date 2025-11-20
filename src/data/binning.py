"""
Binning utilities for ML-based histogram creation.
"""

import numpy as np


def create_histograms_from_binary_scores(dataset, model, features, n_bins=20):
    """
    Build histograms using binary classifier scores.
    
    Args:
        dataset: Dict with keys "Z", "ggH", "VBF" containing DataFrames
        model: Trained binary classifier with predict_proba method
        features: List of feature names to use
        n_bins: Number of bins
    
    Returns:
        histograms: Dict of histograms for each process
        bins: Bin edges
    """
    bins = np.linspace(0, 1, n_bins + 1)
    histograms = {}
    
    for process in ["Z", "ggH", "VBF"]:
        x = dataset[process][features]
        scores = model.predict_proba(x)[:, 1]
        histograms[process] = np.histogram(scores, bins=bins)[0]
    
    return histograms, bins


def create_2d_histograms_from_multiclass(dataset, model, features, n_bins_x=5, n_bins_y=10):
    """
    Build 2D histograms using multiclass scores.
    x-axis: P(ggH) / (P(ggH) + P(VBF))  - discriminates ggH vs VBF
    y-axis: P(ggH) + P(VBF) - discriminates signal vs background
    
    Args:
        dataset: Dict with keys "Z", "ggH", "VBF" containing DataFrames
        model: Trained multiclass classifier with predict_proba method
        features: List of feature names to use
        n_bins_x: Number of bins in x direction
        n_bins_y: Number of bins in y direction
    
    Returns:
        histograms: Dict of flattened 2D histograms for each process
        bins: Tuple of (x_edges, y_edges)
    """
    histograms = {}
    
    for process in ["Z", "ggH", "VBF"]:
        x = dataset[process][features]
        probs = model.predict_proba(x)
        
        signal_prob = probs[:, 1] + probs[:, 2]
        ggH_fraction = np.divide(
            probs[:, 1], 
            signal_prob,
            out=np.zeros_like(signal_prob),
            where=signal_prob != 0
        )
        
        hist, x_edges, y_edges = np.histogram2d(
            ggH_fraction, 
            signal_prob,
            bins=[n_bins_x, n_bins_y],
            range=[[0, 1], [0, 1]]
        )
        
        histograms[process] = hist.flatten()
    
    return histograms, (x_edges, y_edges)
