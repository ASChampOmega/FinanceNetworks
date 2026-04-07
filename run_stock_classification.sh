#!/usr/bin/env bash
# =============================================================================
# run_stock_classification.sh — Stock dataset classification experiment.
#
# 1. Activates the "forecast" conda environment.
# 2. Runs classification.py (volatility-spike classification on 100 US stocks).
#    – Reuses cached graph features from cross_val.py if available.
# 3. Generates ablation plots for stock classification.
# 4. Copies images into stock_dataset_results/.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
FN_DIR="$REPO_ROOT/FinanceNetworks"

# ── 0. Activate conda environment ───────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate forecast
cd "$FN_DIR"

# ── 1. Run experiment ───────────────────────────────────────────────────────
echo "========================================"
echo " Running classification.py (stock classification)"
echo "========================================"
python -m evaluation.classification "$@"

# ── 2. Generate plots ───────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Generating stock classification plots"
echo "========================================"
echo "  -> Ablation plots (stock classification) ..."
python -m visualize.ablation_plot --results-dir results --task classification

echo "  -> Ablation CSV tables (classification) ..."
python -m evaluation.classification_ablation

echo "  -> Writing clean summary log ..."
python visualize/print_results_classification.py > "$REPO_ROOT/log_classification.txt"

# ── 3. Copy images ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Updating stock_dataset_results/"
echo "========================================"

DEST="$REPO_ROOT/stock_dataset_results"
PLOTS="$FN_DIR/results/plots"

mkdir -p "$DEST/ablation"

if [[ -d "$PLOTS/ablation" ]]; then
    cp -f "$PLOTS/ablation/"*.png "$DEST/ablation/" 2>/dev/null || true
fi

echo ""
echo "Done! Log: $REPO_ROOT/log_classification.txt"
