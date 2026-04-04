"""
visualize/print_results_index.py
================================
Load saved index-experiment results and reprint all summary tables / plots.

Mirrors print_results.py's __main__ block but points at
``results/index_results/`` and uses index-appropriate sample tickers.

Usage
-----
    python visualize/print_results_index.py [--metric log|raw] [--selection r2|mean_rmse|median_rmse] [--plot] [--present]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visualize.print_results import load_and_print_results

INDEX_SAMPLE_TICKERS = ["SPX2", "FTSE2", "N2252", "GDAXI2", "IXIC2"]
INDEX_RESULTS_DIR = Path(__file__).parent.parent / "results" / "index_results"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print index regression benchmark results from saved JSON.",
    )
    parser.add_argument(
        "--metric",
        choices=["log", "raw"],
        default="log",
        help=(
            "Priority metric for ranking and selecting 'best' models. "
            "'log' (default) uses RMSE_log / R2_log; "
            "'raw' uses RMSE / R2."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=["r2", "mean_rmse", "median_rmse"],
        default="r2",
        help=(
            "Model selection criterion. "
            "'r2' (default) ranks by pct R2 > 0; "
            "'mean_rmse' ranks by mean RMSE (lower is better); "
            "'median_rmse' ranks by median RMSE (lower is better)."
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
        help="Path to index results directory.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate and save the summary bar chart.",
    )
    parser.add_argument(
        "--present",
        action="store_true",
        help="Generate condensed presentation-ready tables.",
    )
    args = parser.parse_args()
    load_and_print_results(
        results_dir=args.results_dir,
        sample_tickers=args.tickers,
        plot=args.plot,
        use_log=(args.metric == "log"),
        selection=args.selection,
        present=args.present,
    )
