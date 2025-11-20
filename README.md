# Higgs Boson Classification Project

Machine learning-based search for Higgs boson events using binary and multiclass classification to optimise binned likelihood fits.

## Project Structure

```
higgs-boson-classification/
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
│   ├── analysis.ipynb                      # Concise summary of complete analysis
│   ├── exploratory_data_analysis.ipynb     # EDA of Higgs dataset
│   ├── binary_classification.ipynb         # Binary classification experiments
│   └── multiclass_classification.ipynb     # Multiclass classification experiments
├── configs/
│   ├── xgboost_config.yaml                 # XGBoost hyperparameters
│   ├── nn_config.yaml                      # Neural network architecture
│   └── rf_config.yaml                      # Random Forest parameters
├── .github/
│   └── copilot-instructions.md             # AI coding assistant guidelines
├── .gitignore                              # Git ignore rules
└── requirements.txt                        # Python dependencies
```

## Installation

```bash
cd higgs-boson-classification
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Complete Analysis

Open and run `notebooks/analysis.ipynb` for a concise, end-to-end analysis:

1. Loads Higgs boson data (Z, ggH, VBF processes across et, mt, tt channels)
2. Trains multiclass XGBoost classifiers per channel
3. Creates optimised 2D histograms
4. Runs binned likelihood fits
5. Compares results against CMS benchmarks

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
- Optimised binning: 4×15 grid (60 bins) favouring S/B discrimination
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

## Notebooks

1. **`main_analysis.ipynb`** - Concise summary of complete analysis (recommended starting point)
2. `exploratory_data_analysis.ipynb` - Data exploration and visualisation
3. `binary_classification.ipynb` - Binary classification (signal vs background)
4. `multiclass_classification.ipynb` - Multiclass classification with binning optimisation

## Configuration

Hyperparameters can be adjusted in YAML files in `configs/`:

- `xgboost_config.yaml` - Learning rate, max depth, n_estimators, etc.
- `nn_config.yaml` - Hidden layers, activation functions, optimizer
- `rf_config.yaml` - Number of trees, max features, max depth

## Results

Expected improvements over classical methods:

| Method | μ (%) | μ_ggH (%) | μ_VBF (%) |
|--------|-------|-----------|-----------|
| Classical (m_vis) | ~10 | ~57 | ~490 |
| Classical (Optimised) | ~8 | ~12 | ~35 |
| Binary ML | ~6-8 | ~10-12 | ~20-30 |
| Multiclass ML | ~6 | ~9 | ~18 |
| **CMS Published** | **6** | **9** | **18** |

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