#!/usr/bin/env bash
# =============================================================================
# download_data.sh — Download and preprocess all datasets.
#
# 1. Downloads daily OHLCV data for the top 500 S&P 500 stocks via yfinance.
# 2. Downloads SPY (market proxy) data.
# 3. Verifies the Oxford-Man Realized Volatility Indices CSV is present.
#
# Usage:
#   bash download_data.sh              # full 500-stock download
#   bash download_data.sh --n-tickers 100  # only top 100
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
FN_DIR="$REPO_ROOT/FinanceNetworks"
DATA_DIR="$FN_DIR/data/data_files"

N_TICKERS=500

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-tickers)
            N_TICKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash download_data.sh [--n-tickers N]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo " download_data.sh"
echo "========================================"

# ── 0. Activate conda environment ───────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate forecast
cd "$FN_DIR"

# ── 1. Verify sp500_list.txt exists ─────────────────────────────────────────
if [[ ! -f "$DATA_DIR/sp500_list.txt" ]]; then
    echo "ERROR: $DATA_DIR/sp500_list.txt not found."
    echo "This file is required to determine which stocks to download."
    exit 1
fi

# ── 2. Download stock data ──────────────────────────────────────────────────
echo ""
echo "Downloading daily OHLCV data for top $N_TICKERS S&P 500 stocks..."
echo "(Already-downloaded tickers will be skipped.)"
python -c "
from data.load_data import download_data_all, get_market_correlation
print('--- Downloading stock data ---')
download_data_all($N_TICKERS)
print('--- Downloading SPY (market proxy) ---')
get_market_correlation()
print('--- Stock data download complete ---')
"

# ── 3. Count downloaded files ────────────────────────────────────────────────
N_FILES=$(find "$DATA_DIR" -name '*_data.csv' | wc -l)
echo ""
echo "Downloaded data files: $N_FILES CSVs in $DATA_DIR"

# ── 4. Verify Oxford-Man CSV ────────────────────────────────────────────────
OXFORD_CSV="$DATA_DIR/OxfordManRealizedVolatilityIndices.csv"
if [[ -f "$OXFORD_CSV" ]]; then
    echo "Oxford-Man Realized Volatility Indices CSV: found ✓"
else
    echo ""
    echo "WARNING: Oxford-Man CSV not found at:"
    echo "  $OXFORD_CSV"
    echo ""
    echo "This file is required for the index-dataset experiments."
    echo "Download it from: https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip"
    echo "and place the CSV file at the path above."
    exit 1
fi

echo ""
echo "========================================"
echo " Data download complete."
echo "========================================"
