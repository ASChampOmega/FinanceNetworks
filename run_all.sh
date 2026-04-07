#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Master script for the Econ423 FinanceNetworks project.
#
# Runs all 4 experiments in order.  Regression scripts run first so that
# their graph caches are available for the classification scripts (which
# skip the expensive fit_transform when a cache is found).
#
# Each experiment has its own self-contained script:
#   run_stock_regression.sh      -> cross_val.py      + plots
#   run_stock_classification.sh  -> classification.py  + plots (reuses stock graph cache)
#   run_index_regression.sh      -> cross_val_index.py + plots
#   run_index_classification.sh  -> classification_index.py + plots (reuses index graph cache)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo " run_all.sh — Full experiment pipeline"
echo "========================================"

# ── 1. Stock regression (builds + caches graphs) ────────────────────────────
echo ""
echo "[1/4] Stock regression ..."
bash "$REPO_ROOT/run_stock_regression.sh" "$@"

# ── 2. Stock classification (loads cached graphs) ───────────────────────────
echo ""
echo "[2/4] Stock classification ..."
bash "$REPO_ROOT/run_stock_classification.sh" "$@"

# ── 3. Index regression (builds + caches graphs) ────────────────────────────
echo ""
echo "[3/4] Index regression ..."
bash "$REPO_ROOT/run_index_regression.sh" "$@"

# ── 4. Index classification (loads cached graphs) ───────────────────────────
echo ""
echo "[4/4] Index classification ..."
bash "$REPO_ROOT/run_index_classification.sh" "$@"

echo ""
echo "========================================"
echo " All done!"
echo "========================================"
echo "Logs:"
echo "  $REPO_ROOT/log_cross_val.txt"
echo "  $REPO_ROOT/log_cross_val_index.txt"
echo "  $REPO_ROOT/log_classification.txt"
echo "  $REPO_ROOT/log_classification_index.txt"
echo ""
echo "Result images:"
echo "  $REPO_ROOT/stock_dataset_results"
echo "  $REPO_ROOT/index_dataset_results"
