#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — One-command master script for the entire project.
#
# Runs the full pipeline end-to-end:
#   0. Install the FinanceNetworks package
#   1. Download and validate data
#   2. Run all 4 experiments (stock/index × regression/classification)
#   3. Generate plots, ablation tables, and summary logs
#
# Flags:
#   --sanity    Run a fast smoke-test version (~5 min) with fewer tickers
#               and models.  Results go to sanity_results/ so they never
#               overwrite real experiment outputs.
#   --skip-download   Skip the data-download step (data already present).
#   --skip-install    Skip the pip install step.
#
# Usage:
#   bash run_pipeline.sh              # full run
#   bash run_pipeline.sh --sanity     # quick end-to-end smoke test
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
FN_DIR="$REPO_ROOT/FinanceNetworks"

SANITY=0
SKIP_DOWNLOAD=0
SKIP_INSTALL=0

# ── Parse arguments ──────────────────────────────────────────────────────────
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sanity)
            SANITY=1
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        --single-fold)
            PASSTHROUGH_ARGS+=("--single-fold")
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "========================================"
echo " run_pipeline.sh — Full project pipeline"
if [[ "$SANITY" -eq 1 ]]; then
    echo " MODE: sanity check (fast smoke test)"
fi
echo "========================================"

# ── 0. Activate conda environment ───────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate forecast

# ── 1. Install package ──────────────────────────────────────────────────────
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    echo ""
    echo "========================================"
    echo " Installing FinanceNetworks package"
    echo "========================================"
    cd "$FN_DIR"
    pip install -e . --quiet
fi

# ── 2. Download data ────────────────────────────────────────────────────────
if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    echo ""
    echo "========================================"
    echo " Downloading data"
    echo "========================================"
    if [[ "$SANITY" -eq 1 ]]; then
        bash "$REPO_ROOT/download_data.sh" --n-tickers 35
    else
        bash "$REPO_ROOT/download_data.sh"
    fi
fi

# ── 3. Run experiments ──────────────────────────────────────────────────────
if [[ "$SANITY" -eq 1 ]]; then
    # ── Sanity mode: lightweight smoke test ──────────────────────────────
    echo ""
    echo "========================================"
    echo " Running sanity-check experiments"
    echo "========================================"
    SANITY_DIR="$REPO_ROOT/sanity_results"
    mkdir -p "$SANITY_DIR"

    cd "$FN_DIR"

    echo ""
    echo "[sanity 1/4] Stock regression smoke test ..."
    python -m evaluation.sanity_runner \
        --task stock-regression \
        --results-dir "$SANITY_DIR/stock_regression" \
        "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"

    echo ""
    echo "[sanity 2/4] Stock classification smoke test ..."
    python -m evaluation.sanity_runner \
        --task stock-classification \
        --results-dir "$SANITY_DIR/stock_classification" \
        "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"

    echo ""
    echo "[sanity 3/4] Index regression smoke test ..."
    python -m evaluation.sanity_runner \
        --task index-regression \
        --results-dir "$SANITY_DIR/index_regression" \
        "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"

    echo ""
    echo "[sanity 4/4] Index classification smoke test ..."
    python -m evaluation.sanity_runner \
        --task index-classification \
        --results-dir "$SANITY_DIR/index_classification" \
        "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"

    echo ""
    echo "========================================"
    echo " Sanity check complete!"
    echo "========================================"
    echo ""
    echo "Results written to: $SANITY_DIR/"
    echo "These are throwaway smoke-test outputs — not real experiment results."

else
    # ── Full mode: run all 4 experiments ─────────────────────────────────
    echo ""
    echo "========================================"
    echo " Running full experiments"
    echo "========================================"
    bash "$REPO_ROOT/run_all.sh" "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"

    # ── 4. Regenerate presentation logs + ablation plots ─────────────────
    echo ""
    echo "========================================"
    echo " Regenerating summary logs and ablation plots"
    echo "========================================"
    bash "$REPO_ROOT/regenerate_regression_logs.sh"

    echo ""
    echo "========================================"
    echo " Pipeline complete!"
    echo "========================================"
    echo ""
    echo "Key outputs:"
    echo "  Logs:    log_cross_val*.txt, log_classification*.txt"
    echo "  Plots:   stock_dataset_results/, index_dataset_results/"
    echo "  Results: FinanceNetworks/results/"
fi
