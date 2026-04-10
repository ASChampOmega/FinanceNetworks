# Volatility Forecasting with Financial Networks

**ECON 423 Final Project — Aaryam Sharma**

This project forecasts stock and index volatility using network-augmented
econometric models.  It builds correlation networks from rolling windows of
returns, extracts graph-based features (degree centrality, clustering
coefficient, neighborhood volatility, etc.), and feeds them alongside
classical HAR (Heterogeneous AutoRegressive) features into regression and
classification models.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Quick Start](#quick-start)
5. [Datasets](#datasets)
6. [Models](#models)
7. [Experiments](#experiments)
8. [Outputs](#outputs)
9. [Sanity Check](#sanity-check)
10. [Running Individual Experiments](#running-individual-experiments)

---

## Overview

### Research Question

Can inter-stock correlation networks improve volatility forecasting beyond
classical HAR and GARCH baselines?

### Approach

1. **Feature engineering**: Compute daily realized variance (RV1), weekly
   (RV5), bi-weekly (RV10), and monthly (RV22) rolling means, plus leverage
   (semivariance) and market-level features.
2. **Network construction**: Build correlation graphs over rolling 60-day
   windows using three distance metrics — squared correlation, partial
   correlation, and mutual information.  KNN graphs are formed for
   k ∈ {1, 2, 3, 4, 5}.
3. **Graph feature extraction**: For each ticker at each date, extract
   degree centrality, clustering coefficient, inverse-distance-weighted
   neighbor RV features, and optional spectral clustering membership.
4. **Two tasks**:
   - **Regression** — Predict 5-day-ahead realized variance (Y_fwd).
   - **Classification** — Predict volatility spikes (binary: above the
     80th/75th percentile of training-set Y_fwd).
5. **Evaluation**: Expanding-window cross-validation with strict temporal
   separation (no look-ahead leakage).

### Two Datasets

| Dataset | Universe | Source | RV Measure |
|---------|----------|--------|------------|
| **Stock** | Top 100 S&P 500 stocks | Yahoo Finance (daily OHLCV) | Squared daily log-returns |
| **Index** | 21 global equity indices | Custom dataset (provided by course instructor, not publicly available) | 5-minute realized variance |

---

## Project Structure

```
Econ423-FinalProject/
│
├── run_pipeline.sh              # ★ Master script — runs everything
├── download_data.sh             # Downloads stock data via yfinance
├── run_all.sh                   # Runs all 4 experiments (no data download)
├── run_stock_regression.sh      # Stock regression only
├── run_stock_classification.sh  # Stock classification only
├── run_index_regression.sh      # Index regression only
├── run_index_classification.sh  # Index classification only
├── regenerate_regression_logs.sh # Re-generate logs/plots from saved results
│
├── FinanceNetworks/             # Python package (pip install -e .)
│   ├── setup.py
│   ├── data/
│   │   ├── load_data.py         # yfinance download + CSV caching
│   │   ├── preprocess.py        # HAR feature construction (stocks)
│   │   ├── preprocess_index.py  # HAR features from 5-min realized variance indices
│   │   ├── volatility_spikes.py # Binary spike labelling (per-fold)
│   │   ├── graph_cache.py       # Pickle serialization for graph reuse
│   │   └── data_files/          # Raw CSV data (auto-downloaded)
│   │       ├── sp500_list.txt
│   │       ├── OxfordManRealizedVolatilityIndices.csv
│   │       └── *_data.csv       # Per-ticker OHLCV files
│   │
│   ├── models/
│   │   ├── baselines.py                   # HAR, ARIMA, GARCH, DCC-GARCH, 2-regime HAR
│   │   ├── baselines_classification.py    # Logistic versions of the above
│   │   ├── network_models.py              # NetworkHAR, NetworkVAR, LearnedWeightNetworkHAR
│   │   ├── network_models_classification.py # Classification versions
│   │   └── correlation_network.py         # SquaredCorr, PartialCorr, MutualInfo networks
│   │
│   ├── evaluation/
│   │   ├── cross_val.py             # Stock regression CV
│   │   ├── cross_val_index.py       # Index regression CV
│   │   ├── classification.py        # Stock classification CV
│   │   ├── classification_index.py  # Index classification CV
│   │   ├── forecasting_ablation.py  # Regression ablation tables
│   │   ├── classification_ablation.py # Classification ablation tables
│   │   ├── interpretability.py      # Model param / graph snapshot export
│   │   └── sanity_runner.py         # Lightweight smoke-test runner
│   │
│   ├── visualize/
│   │   ├── plot_model_results.py          # Stock prediction & summary plots
│   │   ├── plot_model_results_index.py    # Index prediction & summary plots
│   │   ├── print_results.py              # Regression text summaries
│   │   ├── print_results_index.py        # Index regression summaries
│   │   ├── print_results_classification.py       # Stock classification summaries
│   │   ├── print_results_classification_index.py # Index classification summaries
│   │   ├── ablation_plot.py               # Ablation comparison plots
│   │   └── data_distribution.py           # EDA plots
│   │
│   └── results/                 # Experiment outputs (auto-generated)
│       ├── results_bench.json
│       ├── summary_bench.json
│       ├── classification_results.json
│       ├── classification_summary.json
│       ├── graphs/              # Saved graph snapshots
│       ├── graph_cache/         # Pickled graph features for reuse
│       ├── feature_snapshots/   # Per-ticker feature matrices
│       ├── predictions_regression/
│       ├── predictions_classification/
│       ├── plots/               # All generated figures
│       ├── ablation/            # Ablation CSV tables
│       └── index_results/       # Mirror of above for index dataset
│
├── stock_dataset_results/       # Copied plots for stock experiments
├── index_dataset_results/       # Copied plots for index experiments
├── sanity_results/              # Throwaway outputs from --sanity runs
├── data_nb.ipynb                # Exploratory data notebook
└── log_*.txt                    # Summary log files
```

---

## Setup

### Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Python 3.7+**
- **Linux or macOS** (uses fork-based multiprocessing)

### 1. Create the conda environment

```bash
conda create -n forecast python=3.10 -y
conda activate forecast
```

### 2. Install the package and dependencies

```bash
cd FinanceNetworks
pip install -e .
```

This installs: numpy, pandas, matplotlib, scipy, yfinance, scikit-learn,
arch, tqdm, statsmodels, networkx.

### 3. Prepare data

The stock data is downloaded automatically from Yahoo Finance.  The index
dataset (`OxfordManRealizedVolatilityIndices.csv`) was provided by the course
instructor and is **not publicly available** — place it manually at:

```
FinanceNetworks/data/data_files/OxfordManRealizedVolatilityIndices.csv
```

To download just the stock data:

```bash
bash download_data.sh                # all 500 tickers
bash download_data.sh --n-tickers 100  # top 100 only
```

---

## Quick Start

### Run everything (data download → experiments → plots → logs)

```bash
bash run_pipeline.sh
```

### Run a fast smoke test first (~5 minutes)

```bash
bash run_pipeline.sh --sanity
```

This runs a minimal version of all 4 experiments (2 tickers, 1 k-value,
~4 models each) and writes results to `sanity_results/`.  Use this to verify
the full pipeline works before committing to a multi-hour run.

### Skip data download (if data is already present)

```bash
bash run_pipeline.sh --skip-download
```

---

## Datasets

### Stock Dataset (100 S&P 500 Stocks)

- **Source**: Yahoo Finance daily OHLCV (2015–2026)
- **Target**: 5-day forward realized variance Y_fwd = Σ r²_{t+1..t+5}
- **Features**: RV1, RV5, RV10, RV22, semivariance (neg/pos), market RV,
  log transforms, and network-derived features
- **Classification threshold**: 80th percentile of training-set Y_fwd

### Index Dataset (21 Global Indices)

- **Source**: Custom dataset provided by the course instructor (not publicly available)
- **Indices**: SPX2, FTSE2, N2252, GDAXI2, RUT2, AORD2, DJI2, IXIC2,
  FCHI2, HSI2, KS11, AEX, SSMI, IBEX2, NSEI, MXX, BVSP, GSPTSE,
  STOXX50E, FTSTI, FTSEMIB
- **Market proxy**: SPX2 (S&P 500)
- **Classification threshold**: 75th percentile

---

## Models

### Baselines (Regression)

| Model | Description |
|-------|-------------|
| **HAR** | Heterogeneous AutoRegressive — linear in RV1, RV5, RV22 |
| **HAR-Extended** | Adds semivariance + market features |
| **ARIMA** | Various (p,d,q) orders on log(Y_fwd) |
| **GARCH** | GARCH(p,q) on daily returns → weekly RV forecast |
| **DCC-GARCH** | Dynamic Conditional Correlation GARCH |
| **2-Regime HAR** | Markov regime-switching HAR |

### Network Models (Regression)

| Model | Description |
|-------|-------------|
| **NetworkHAR** | Single-stage HAR + graph features (Lasso/Ridge) |
| **NetworkVAR** | Two-stage: HAR first → network residual correction |
| **LearnedWeightNetworkHAR** | SVD-projected neighbor features |

### Classification

Logistic variants of all the above, evaluated by ROC-AUC, accuracy,
precision, and recall.

### Network Types

| Distance Metric | Description |
|-----------------|-------------|
| Squared Correlation | Pairwise r² from rolling returns |
| Partial Correlation | Ledoit-Wolf shrinkage → partial correlations |
| Mutual Information | Binned MI from rolling return windows |

Each metric is used with both inverse-distance and exponential weighting
kernels, and optional spectral clustering.

---

## Experiments

The project runs 4 experiments:

| # | Script | Description |
|---|--------|-------------|
| 1 | `run_stock_regression.sh` | Expanding-window CV on 100 stocks (regression) |
| 2 | `run_stock_classification.sh` | Expanding-window CV on 100 stocks (classification) |
| 3 | `run_index_regression.sh` | Expanding-window CV on 21 indices (regression) |
| 4 | `run_index_classification.sh` | Expanding-window CV on 21 indices (classification) |

Regression must run before classification for each dataset because the
classification step reuses cached graph features from the regression step.

---

## Outputs

### Result Files

| File | Contents |
|------|----------|
| `results/results_bench.json` | Per-ticker, per-model, per-fold regression metrics |
| `results/summary_bench.json` | Aggregated regression summary |
| `results/classification_results.json` | Classification metrics |
| `results/classification_summary.json` | Classification summary |
| `results/index_results/` | Same structure for index dataset |

### Log Files (human-readable summaries)

| File | Contents |
|------|----------|
| `log_cross_val.txt` | Stock regression summary |
| `log_cross_val_index.txt` | Index regression summary |
| `log_classification.txt` | Stock classification summary |
| `log_classification_index.txt` | Index classification summary |
| `log_*_present.txt` | Presentation-style versions (subset of tickers) |

### Plots

- `stock_dataset_results/` — prediction plots, ablation plots
- `index_dataset_results/` — prediction plots, ablation plots
- `FinanceNetworks/results/plots/` — all generated figures

---

## Sanity Check

To quickly verify the pipeline works end-to-end without waiting hours:

```bash
bash run_pipeline.sh --sanity
```

This runs:
- Only 2 tickers per dataset (AAPL + MSFT for stocks, SPX2 + FTSE2 for indices)
- Only k=3 (instead of k=1..5)
- Only 1 graph type (squared correlation)
- ~4 models per experiment (instead of 200+)
- 1 CV fold

Results are written to `sanity_results/` (throwaway). If all 4 tasks
print "PASSED ✓", the codebase is working correctly.

---

## Running Individual Experiments

Each experiment can be run independently:

```bash
# Activate environment first
conda activate forecast

# Stock regression (builds graph cache)
bash run_stock_regression.sh

# Stock classification (reuses cached graphs)
bash run_stock_classification.sh

# Index regression (builds graph cache)
bash run_index_regression.sh

# Index classification (reuses cached graphs)
bash run_index_classification.sh
```

To regenerate logs and ablation plots from existing results without
re-running experiments:

```bash
bash regenerate_regression_logs.sh
```
