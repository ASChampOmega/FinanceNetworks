#!/usr/bin/env bash
# =============================================================================
# run_stock_regression.sh — Stock dataset regression experiment.
#
# 1. Activates the "forecast" conda environment.
# 2. Runs cross_val.py (expanding-window CV on 100 US stocks).
# 3. Generates prediction/summary plots and ablation plots.
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
echo " Running cross_val.py (stock regression)"
echo "========================================"
python -m evaluation.cross_val "$@"

# ── 2. Generate plots ───────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Generating stock regression plots"
echo "========================================"
echo "  -> Prediction + summary plots ..."
python visualize/plot_model_results.py

echo "  -> Ablation plots (stock regression) ..."
python -m visualize.ablation_plot --results-dir results --task regression

echo "  -> Ablation CSV tables ..."
python -m evaluation.forecasting_ablation

echo "  -> Writing clean summary log ..."
python visualize/print_results.py > "$REPO_ROOT/log_cross_val.txt"

# ── 3. Copy images ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Updating stock_dataset_results/"
echo "========================================"

DEST="$REPO_ROOT/stock_dataset_results"
PLOTS="$FN_DIR/results/plots"

mkdir -p "$DEST/predictions_log" "$DEST/predictions_raw" "$DEST/ablation"

if [[ -d "$PLOTS/predictions_log" ]]; then
    cp -f "$PLOTS/predictions_log/"*.png "$DEST/predictions_log/" 2>/dev/null || true
fi
if [[ -d "$PLOTS/predictions_raw" ]]; then
    cp -f "$PLOTS/predictions_raw/"*.png "$DEST/predictions_raw/" 2>/dev/null || true
fi
cp -f "$PLOTS/"*.png "$DEST/" 2>/dev/null || true
if [[ -d "$PLOTS/ablation" ]]; then
    cp -f "$PLOTS/ablation/"*.png "$DEST/ablation/" 2>/dev/null || true
fi

echo ""
echo "Done! Log: $REPO_ROOT/log_cross_val.txt"
