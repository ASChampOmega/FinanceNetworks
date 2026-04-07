#!/usr/bin/env bash
# =============================================================================
# run_index_regression.sh — Index dataset regression experiment.
#
# 1. Activates the "forecast" conda environment.
# 2. Runs cross_val_index.py (expanding-window CV on 21 global indices).
# 3. Generates prediction/summary plots and ablation plots.
# 4. Copies images into index_dataset_results/.
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
echo " Running cross_val_index.py (index regression)"
echo "========================================"
python -m evaluation.cross_val_index "$@"

# ── 2. Generate plots ───────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Generating index regression plots"
echo "========================================"
echo "  -> Prediction + summary plots ..."
python visualize/plot_model_results_index.py

echo "  -> Ablation plots (index regression) ..."
python -m visualize.ablation_plot --results-dir results/index_results --task regression

echo "  -> Writing clean summary log ..."
python visualize/print_results_index.py > "$REPO_ROOT/log_cross_val_index.txt"

# ── 3. Copy images ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Updating index_dataset_results/"
echo "========================================"

DEST="$REPO_ROOT/index_dataset_results"
PLOTS="$FN_DIR/results/index_results/plots"

mkdir -p "$DEST/plots/predictions_log" "$DEST/plots/ablation"

if [[ -d "$PLOTS/predictions_log" ]]; then
    cp -f "$PLOTS/predictions_log/"*.png "$DEST/plots/predictions_log/" 2>/dev/null || true
fi
cp -f "$PLOTS/"*.png "$DEST/plots/" 2>/dev/null || true
if [[ -d "$PLOTS/ablation" ]]; then
    cp -f "$PLOTS/ablation/"*.png "$DEST/plots/ablation/" 2>/dev/null || true
fi

echo ""
echo "Done! Log: $REPO_ROOT/log_cross_val_index.txt"
