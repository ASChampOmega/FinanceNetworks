#!/usr/bin/env bash

# Generates presentation-style summary logs and ablation plots from saved
# regression + classification results without rerunning any experiments.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
FN_DIR="$REPO_ROOT/FinanceNetworks"

STOCK_REG_LOG="$REPO_ROOT/log_cross_val_present.txt"
INDEX_REG_LOG="$REPO_ROOT/log_cross_val_index_present.txt"
STOCK_CLF_LOG="$REPO_ROOT/log_classification_present.txt"
INDEX_CLF_LOG="$REPO_ROOT/log_classification_index_present.txt"

echo "========================================"
echo " regenerate_regression_logs.sh"
echo "========================================"

eval "$(conda shell.bash hook)"
conda activate forecast
cd "$FN_DIR"

echo ""
echo "========================================"
echo " Regenerating regression and classification ablation plots"
echo "========================================"
python -m visualize.ablation_plot \
    --results-dir results \
    --results-dir results/index_results \
    --task both

echo ""
echo "========================================"
echo " Regenerating stock regression logs"
echo "========================================"
python visualize/print_results.py \
    --results-dir results \
    --present-only \
    --tickers AAPL TSLA GOOG META MSFT NVDA NFLX AMZN \
    > "$STOCK_REG_LOG"

echo ""
echo "========================================"
echo " Regenerating index regression logs"
echo "========================================"
python visualize/print_results_index.py \
    --results-dir results/index_results \
    --present-only \
    > "$INDEX_REG_LOG"

echo ""
echo "========================================"
echo " Regenerating stock classification logs"
echo "========================================"
python visualize/print_results_classification.py \
    --results-dir results \
    --present-only \
    --tickers AAPL TSLA GOOG META MSFT NVDA NFLX AMZN \
    > "$STOCK_CLF_LOG"

echo ""
echo "========================================"
echo " Regenerating index classification logs"
echo "========================================"
python visualize/print_results_classification_index.py \
    --results-dir results/index_results \
    --present-only \
    > "$INDEX_CLF_LOG"

echo ""
echo "========================================"
echo " Syncing ablation plots to result folders"
echo "========================================"

mkdir -p "$REPO_ROOT/stock_dataset_results/ablation"
mkdir -p "$REPO_ROOT/index_dataset_results/plots/ablation"

cp -f "$FN_DIR/results/plots/ablation/"*.png "$REPO_ROOT/stock_dataset_results/ablation/" 2>/dev/null || true
cp -f "$FN_DIR/results/index_results/plots/ablation/"*.png "$REPO_ROOT/index_dataset_results/plots/ablation/" 2>/dev/null || true

echo ""
echo "Done. Updated files:"
echo "  $STOCK_REG_LOG"
echo "  $INDEX_REG_LOG"
echo "  $STOCK_CLF_LOG"
echo "  $INDEX_CLF_LOG"