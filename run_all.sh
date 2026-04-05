#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Master script for the Econ423 FinanceNetworks project.
#
# 1. Activates the "forecast" conda environment.
# 2. Runs each evaluation experiment:
#      cross_val, cross_val_index, classification, classification_index
# 3. Regenerates all plots (prediction, summary, ablation).
# 4. Copies updated images into index_dataset_results/ and
#    stock_dataset_results/.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
FN_DIR="$REPO_ROOT/FinanceNetworks"

# ── 0. Activate conda environment ───────────────────────────────────────────
echo "========================================"
echo " Activating conda env: forecast"
echo "========================================"
# Source conda's shell hooks so 'conda activate' works in scripts
eval "$(conda shell.bash hook)"
conda activate forecast

cd "$FN_DIR"

# ── 1. Run regression on stock dataset ───────────────────────────────────────
echo ""
echo "========================================"
echo " [1/4] Running cross_val.py (stock regression)"
echo "========================================"
python -m evaluation.cross_val 2>&1 | tee "$REPO_ROOT/log_cross_val.txt"

# ── 2. Run regression on index dataset ───────────────────────────────────────
echo ""
echo "========================================"
echo " [2/4] Running cross_val_index.py (index regression)"
echo "========================================"
python -m evaluation.cross_val_index 2>&1 | tee "$REPO_ROOT/log_cross_val_index.txt"

# ── 3. Run classification on stock dataset ───────────────────────────────────
echo ""
echo "========================================"
echo " [3/4] Running classification.py (stock classification)"
echo "========================================"
python -m evaluation.classification 2>&1 | tee "$REPO_ROOT/log_classification.txt"

# ── 4. Run classification on index dataset ───────────────────────────────────
echo ""
echo "========================================"
echo " [4/4] Running classification_index.py (index classification)"
echo "========================================"
python -m evaluation.classification_index 2>&1 | tee "$REPO_ROOT/log_classification_index.txt"

# ── 5. Regenerate all plots ──────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Regenerating plots"
echo "========================================"

# 5a. Stock regression prediction + summary plots
echo "  -> Stock regression plots ..."
python visualize/plot_model_results.py

# 5b. Index regression prediction + summary plots
echo "  -> Index regression plots ..."
python visualize/plot_model_results_index.py

# 5c. Ablation plots (stock + index, regression + classification)
echo "  -> Ablation plots (all tasks, all datasets) ..."
python -m visualize.ablation_plot

# 5d. Ablation CSV tables (regression + classification)
echo "  -> Ablation CSV tables (regression) ..."
python -m evaluation.forecasting_ablation
echo "  -> Ablation CSV tables (classification) ..."
python -m evaluation.classification_ablation

# 5e. Print results (for log reference)
echo "  -> Printing stock regression results ..."
python visualize/print_results.py
echo "  -> Printing index regression results ..."
python visualize/print_results_index.py
echo "  -> Printing stock classification results ..."
python visualize/print_results_classification.py
echo "  -> Printing index classification results ..."
python visualize/print_results_classification_index.py

# ── 6. Copy updated images into top-level result directories ────────────────
echo ""
echo "========================================"
echo " Updating result image directories"
echo "========================================"

STOCK_DEST="$REPO_ROOT/stock_dataset_results"
INDEX_DEST="$REPO_ROOT/index_dataset_results"

STOCK_PLOTS="$FN_DIR/results/plots"
INDEX_PLOTS="$FN_DIR/results/index_results/plots"

# --- Stock dataset results ---
mkdir -p "$STOCK_DEST/predictions_log"
mkdir -p "$STOCK_DEST/predictions_raw"
mkdir -p "$STOCK_DEST/ablation"

# Prediction plots
if [[ -d "$STOCK_PLOTS/predictions_log" ]]; then
    cp -f "$STOCK_PLOTS/predictions_log/"*.png "$STOCK_DEST/predictions_log/" 2>/dev/null || true
fi
if [[ -d "$STOCK_PLOTS/predictions_raw" ]]; then
    cp -f "$STOCK_PLOTS/predictions_raw/"*.png "$STOCK_DEST/predictions_raw/" 2>/dev/null || true
fi

# Summary and neighbourhood plots
cp -f "$STOCK_PLOTS/"*.png "$STOCK_DEST/" 2>/dev/null || true

# Ablation plots
if [[ -d "$STOCK_PLOTS/ablation" ]]; then
    cp -f "$STOCK_PLOTS/ablation/"*.png "$STOCK_DEST/ablation/" 2>/dev/null || true
fi

# --- Index dataset results ---
mkdir -p "$INDEX_DEST/plots/predictions_log"
mkdir -p "$INDEX_DEST/plots/ablation"

# Prediction plots
if [[ -d "$INDEX_PLOTS/predictions_log" ]]; then
    cp -f "$INDEX_PLOTS/predictions_log/"*.png "$INDEX_DEST/plots/predictions_log/" 2>/dev/null || true
fi

# Summary and neighbourhood plots
cp -f "$INDEX_PLOTS/"*.png "$INDEX_DEST/plots/" 2>/dev/null || true

# Ablation plots
if [[ -d "$INDEX_PLOTS/ablation" ]]; then
    cp -f "$INDEX_PLOTS/ablation/"*.png "$INDEX_DEST/plots/ablation/" 2>/dev/null || true
fi

echo ""
echo "========================================"
echo " All done!"
echo "========================================"
echo "Logs saved to:"
echo "  $REPO_ROOT/log_cross_val.txt"
echo "  $REPO_ROOT/log_cross_val_index.txt"
echo "  $REPO_ROOT/log_classification.txt"
echo "  $REPO_ROOT/log_classification_index.txt"
echo ""
echo "Result images updated in:"
echo "  $STOCK_DEST"
echo "  $INDEX_DEST"
