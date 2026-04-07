#!/usr/bin/env bash
# =============================================================================
# run_index_classification.sh — Index dataset classification experiment.
#
# 1. Activates the "forecast" conda environment.
# 2. Runs classification_index.py (volatility-spike classification on 21 indices).
#    – Reuses cached graph features from cross_val_index.py if available.
# 3. Generates ablation plots for index classification.
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
echo " Running classification_index.py (index classification)"
echo "========================================"
python -m evaluation.classification_index "$@"

# ── 2. Generate plots ───────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Generating index classification plots"
echo "========================================"
echo "  -> Ablation plots (index classification) ..."
python -m visualize.ablation_plot --results-dir results/index_results --task classification

echo "  -> Writing clean summary log ..."
python visualize/print_results_classification_index.py > "$REPO_ROOT/log_classification_index.txt"

# ── 3. Copy images ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Updating index_dataset_results/"
echo "========================================"

DEST="$REPO_ROOT/index_dataset_results"
PLOTS="$FN_DIR/results/index_results/plots"

mkdir -p "$DEST/plots/ablation"

if [[ -d "$PLOTS/ablation" ]]; then
    cp -f "$PLOTS/ablation/"*.png "$DEST/plots/ablation/" 2>/dev/null || true
fi

echo ""
echo "Done! Log: $REPO_ROOT/log_classification_index.txt"
