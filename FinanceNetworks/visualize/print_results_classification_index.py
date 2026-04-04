"""
visualize/print_results_classification_index.py
================================================
Load saved index-classification results and reprint all summary tables.

Mirrors print_results_classification.py's CLI entry point but points at
``results/index_results/`` and uses index-appropriate sample tickers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visualize.print_results_classification import load_and_print_classification_results

INDEX_SAMPLE_TICKERS = ["SPX2", "FTSE2", "N2252", "GDAXI2", "IXIC2"]
INDEX_RESULTS_DIR = Path(__file__).parent.parent / "results" / "index_results"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print index classification results from saved JSON.",
    )
    parser.add_argument(
        "--selection",
        choices=["roc_auc", "f1", "weighted_recall"],
        default="roc_auc",
        help=(
            "Metric used for ranking and selecting 'best' models. "
            "'roc_auc' (default) ranks by mean ROC-AUC; "
            "'f1' ranks by mean F1; 'weighted_recall' ranks by mean Weighted Recall."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=INDEX_SAMPLE_TICKERS,
        metavar="TICKER",
        help="Sample tickers for per-ticker tables (default: SPX2 FTSE2 N2252 GDAXI2 IXIC2).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(INDEX_RESULTS_DIR),
        metavar="DIR",
        help="Path to the index results directory.",
    )
    parser.add_argument(
        "--present",
        action="store_true",
        help=(
            "Generate condensed presentation-ready tables: per-ticker top/bottom-5 "
            "improvement, named-ticker table, Wilcoxon test, and structural "
            "comparison tables (distance / weighting / structure)."
        ),
    )
    args = parser.parse_args()
    load_and_print_classification_results(
        results_dir=args.results_dir,
        sample_tickers=args.tickers,
        selection=args.selection,
        present=args.present,
    )