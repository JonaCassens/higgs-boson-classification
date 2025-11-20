# Higgs Boson Classification Project

Machine learning-based search for Higgs boson events using binary and multiclass classification to optimise binned likelihood fits.

## Project Structure

```
higgs-boson-classification/
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── MResFindingTheHiggsAssignment.ipynb     # Original assignment specification
├── src/
│   ├── data/
│   │   ├── loader.py                       # Data loading utilities
│   │   ├── preprocessor.py                 # Data preprocessing and scaling
│   │   └── binning.py                      # ML-based histogram creation
│   └── models/
│       ├── xgboost_classifier.py           # XGBoost classifier (auto-detects binary/multiclass)
│       ├── random_forest.py                # Random Forest classifier
│       └── neural_network.py               # PyTorch neural network
├── notebooks/
│   ├── analysis.ipynb                      # Written summary of complete analysis
│   ├── exploratory_data_analysis.ipynb     # EDA of Higgs dataset
│   ├── binary_classification.ipynb         # Binary classification experiments
│   └── multiclass_classification.ipynb     # Multiclass classification experiments
└── configs/
    ├── xgboost_config.yaml                 # XGBoost hyperparameters
    ├── nn_config.yaml                      # Neural network architecture
    └── rf_config.yaml                      # Random Forest parameters
```

## Installation

```bash
cd higgs-boson-classification
pip install -r requirements.txt
```

## Acknowledgements

AI assistance used for:
- Formatting tables and data for visual clarity
- Markdown cells and headers for clearer separation of sections
- Setting up workspace structure (configs, notebooks, src directories)
- Initial README documentation
- requirements.txt generation


## Quick Start

### Option 1: Read Project Summary

Open `notebooks/analysis.ipynb` for a written summary of the complete project:

- Overview of exploratory data analysis, binary and multiclass classification approaches
- Explanation of methods: data cleaning, model training, probability binning, likelihood fitting
- Summary of results and performance across all notebooks
- No code execution required - just a narrative walkthrough of the analysis

### Option 2: Detailed Exploration

Explore individual notebooks:
- `exploratory_data_analysis.ipynb` - Understand the dataset
- `binary_classification.ipynb` - Signal vs background approach
- `multiclass_classification.ipynb` - Full multiclass analysis with optimisation

## Assignment Goal

Optimise a Higgs boson search by creating ML-based bins that provide the most precise binned likelihood fit for signal rate measurements.

**Target Precision (CMS Published):**
- μ (combined): 6%
- μ_ggH: 9%
- μ_VBF: 18%

## Approach

### Binary Classification
- Separate signal (ggH + VBF) from background (Z)
- Use classifier scores to create 1D histogram bins
- Run likelihood fits for μ_ggH and μ_VBF

### Multiclass Classification (Primary Approach)
- Train **separate XGBoost models per channel** (et, mt, tt)
- Classify events as Z (background), ggH (signal), or VBF (signal)
- Create 2D histograms using multiclass probabilities:
  - x-axis: P(ggH) / [P(ggH) + P(VBF)] — separates production modes
  - y-axis: P(ggH) + P(VBF) — separates signal from background
- Optimised binning: 4×15 grid (60 bins total)
- Combine histograms across channels for final likelihood fit
- Provides superior separation for measuring independent signal strengths

## Models

- **XGBoost**: Gradient boosted trees (primary classifier)
- **Random Forest**: Ensemble of decision trees
- **Neural Network**: PyTorch-based MLP

## Data

Dataset from CMS H→ττ analysis:
- **Channels**: et (electron-tau), mt (muon-tau), tt (tau-tau)
- **Processes**: Z (background), ggH (signal), VBF (signal)
- **Features**: 27 kinematic variables (pt, eta, mass, etc.)



## Configuration

Hyperparameters can be adjusted in YAML files in `configs/`:

- `xgboost_config.yaml` - Learning rate, max depth, n_estimators, etc.
- `nn_config.yaml` - Hidden layers, activation functions, optimizer
- `rf_config.yaml` - Number of trees, max features, max depth

## Results

Precision achieved on signal strength measurements (lower is better):

| Method | μ (%) | μ_ggH (%) | μ_VBF (%) |
|--------|-------|-----------|-----------|  
| **CMS Published** | **6.0** | **9.0** | **18.0** |
| Multiclass XGBoost | 5.78 | 8.47 | 17.22 |

The multiclass approach with per-channel XGBoost models meets all CMS precision benchmarks.

## Key Features

- **Per-channel training**: Separate models preserve channel-specific kinematics
- **Auto-detection**: XGBoost classifier automatically detects binary vs multiclass mode
- **Missing value handling**: Converts -9999 indicators to median-filled values
- **Histogram scaling**: Applies physics cross-section scaling (Z×8.4, ggH×0.034, VBF×0.011)
- **Binning optimisation**: Configurable grid search for optimal precision
- **Likelihood fitting**: Poisson NLL fits using iminuit

## References

- CMS Collaboration H→ττ analysis
- ML Assessment 3 dataset: https://github.com/gputtley/ML-Assessment-3

## License

Academic project for ML course assignment.