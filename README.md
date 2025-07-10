# Financial Time Series Analysis & Trading Simulation using LSTM

This notebook demonstrates a complete workflow for analyzing financial time series data, building an LSTM model to predict trading signals (buy, sell, hold), and backtesting a trading strategy based on those signals.

## Table of Contents

- [Introduction](#introduction)  
- [Setup and Initialization](#setup-and-initialization)  
- [Data Preprocessing](#data-preprocessing)  
  - [Custom Transformers](#custom-transformers)  
  - [Data Loading and Formatting](#data-loading-and-formatting)  
  - [Feature Engineering](#feature-engineering)  
  - [Target Categorization](#target-categorization)  
  - [Preprocessing Pipeline](#preprocessing-pipeline)  
  - [Window Creation](#window-creation)  
- [Dimensionality Reduction](#dimensionality-reduction)  
  - [PCA](#pca)  
  - [t-SNE](#t-sne)  
  - [UMAP](#umap)  
- [Exploratory Data Analysis & Visualization](#exploratory-data-analysis--visualization)  
- [Model Definition & Training](#model-definition--training)  
  - [LSTM Model Architecture](#lstm-model-architecture)  
  - [Training Process](#training-process)  
- [Model Evaluation](#model-evaluation)  
- [Trading Simulation](#trading-simulation)  
  - [Simulation Logic](#simulation-logic)  
  - [Grid Search for Trading Parameters](#grid-search-for-trading-parameters)  
- [Grid Search for Model Hyperparameters & Thresholds](#grid-search-for-model-hyperparameters--thresholds)  
- [Results Analysis](#results-analysis)  
- [Conclusion](#conclusion)  

---

## Introduction

This notebook provides a framework for developing a data-driven trading strategy. It covers:

- Data acquisition and cleaning  
- Feature engineering with technical indicators  
- Building and training an LSTM for signal prediction  
- Backtesting via a trading simulation  

## Setup and Initialization

Import libraries for:

- Data manipulation (`pandas`, `numpy`)  
- Machine learning (`tensorflow`/Keras)  
- Technical analysis (`ta`)  
- Visualization (`matplotlib`, `seaborn`)  

Check GPU availability and configure for optimized training.

## Data Preprocessing

### Custom Transformers

Define scikit-learn compatible transformers:

- **IQRBasedOutlierRemover:** Removes outliers via interquartile range  
- **SMAImputer:** Imputes missing values with simple moving average  
- **FillImputer:** Forward/backward fills missing entries  

### Data Loading and Formatting

Load CSV data, convert price columns to floats, and standardize timestamps as `datetime`.

### Feature Engineering

Add technical indicators using the `ta` library:

- **Momentum:** RSI, Stochastic Oscillator, ROC  
- **Volume:** MFI, Mass Index  
- **Volatility:** ATR  
- **Trend:** CCI, MACD, ADX, Vortex  
- **Others:** DPO, KST, Parabolic SAR, Force Index  
- Compute log returns and wick lengths.

### Target Categorization

Generate trading signals (`buy`, `sell`, `hold`) based on future log returns and threshold rules.

### Preprocessing Pipeline

Use a `ColumnTransformer` to apply:

- Outlier removal  
- Imputation  
- Scaling  

to appropriate feature groups (price, volume, indicators).

### Window Creation

Implement `create_windows` to build sliding windows of features and targets for sequential LSTM training.

## Dimensionality Reduction

### PCA

Perform Principal Component Analysis to examine explained variance and visualize linear projections.

### t-SNE

Use t-Distributed Stochastic Neighbor Embedding for non-linear 2D visualizations, highlighting potential clusters.

### UMAP

Apply Uniform Manifold Approximation and Projection to preserve local and global data structure in low dimensions.

## Exploratory Data Analysis & Visualization

Provide helper functions to plot:

- **Price series & log returns**  
- **Missing-value percentages**  
- **Return distributions with thresholds**  
- **Class balance (buy/sell/hold)**  
- **Feature correlation heatmap**  
- **Feature distributions before/after scaling**  
- **Sample training windows**  

## Model Definition & Training

### LSTM Model Architecture

Define a multi-layer Keras `Sequential` model with:

- Stacked LSTM layers  
- Dropout regularization  
- Dense + softmax output for three-class classification  

### Training Process

Implement `train_model` using time-series cross-validation:

- EarlyStopping & ReduceLROnPlateau callbacks  
- Class-weight computation for imbalance  
- Fold-based splits into train/validation/test  
- TensorFlow `Dataset` pipelines  
- Track per-fold F1 scores and aggregate history  

## Model Evaluation

Use `test_holdoutset` to assess final model on a separate hold-out set:

- Compute accuracy, precision, recall, F1  
- Plot the confusion matrix  

## Trading Simulation

### Simulation Logic

Define `simulate_trades` to:

- Execute trades based on predicted signals  
- Apply transaction costs, take-profit, and stop-loss rules  
- Track capital, wins/losses, and trade entry/exit points  

### Grid Search for Trading Parameters

Implement `grid_search_simulate_trades` to sweep take-profit and stop-loss values, identifying the combination that maximizes final capital.

## Grid Search for Model Hyperparameters & Thresholds

Use `grid_search` to explore:

- Different future-window sizes (`n_future`)  
- Buy/sell threshold values  

Persist progress in a pickle for reproducibility.

## Results Analysis

Provide `find_best_models` to:

- Load grid-search results  
- Identify top models by chosen metric (e.g., final capital)  
- Plot aggregated training histories  

## Conclusion

Although this LSTM-based strategy did not yield a profitable backtest here, the end-to-end process—from data prep and model building to trading simulation—offers a solid foundation for future experimentation with alternative models, features, and tactics.  
