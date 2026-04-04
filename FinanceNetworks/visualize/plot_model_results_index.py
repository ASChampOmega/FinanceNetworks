"""
visualize/plot_model_results_index.py
=====================================
Generate prediction plots and graph-neighbourhood plots for the Oxford-Man
index experiments.

Mirrors plot_model_results.py's CLI entry point but points at
``results/index_results/`` and uses index-appropriate tickers.

Usage
-----
    python visualize/plot_model_results_index.py [--metric log|raw] [--selection r2|mean_rmse|median_rmse]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from visualize.plot_model_results import (
    plot_neighbourhood_change,
    plot_summary_metrics,
    plot_ticker_predictions,
)
from visualize.print_results import summarize_benchmarks

INDEX_SAMPLE_TICKERS = ["SPX2", "FTSE2", "N2252", "GDAXI2", "IXIC2"]
INDEX_RESULTS_DIR = Path(__file__).parent.parent / "results" / "index_results"
INDEX_RAW_DISPLAY_SCALE = 10_000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots for index regression benchmark results.",
    )
    parser.add_argument(
        "--metric",
        choices=["log", "raw"],
        default="log",
        help=(
            "Metric used when selecting best models for prediction plots and summary ordering. "
            "'log' (default) uses RMSE_log / R2_log; 'raw' uses RMSE / R2."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=["r2", "mean_rmse", "median_rmse"],
        default="r2",
        help=(
            "Model selection criterion. 'r2' (default) ranks by pct R2 > 0; "
            "'mean_rmse' ranks by mean RMSE; 'median_rmse' ranks by median RMSE."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=INDEX_SAMPLE_TICKERS,
        metavar="TICKER",
        help="Tickers to use for prediction and neighbourhood plots.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(INDEX_RESULTS_DIR),
        metavar="DIR",
        help="Path to the index results directory.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Skip per-ticker prediction plots.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip the summary bar chart.",
    )
    parser.add_argument(
        "--skip-neighbourhoods",
        action="store_true",
        help="Skip graph neighbourhood-change plots.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    use_log = args.metric == "log"
    metric_tag = "log" if use_log else "raw"

    bench_path = results_dir / "results_bench.json"
    if not bench_path.exists():
        print(f"ERROR: {bench_path} not found. Run evaluation/cross_val_index.py first.")
        sys.exit(1)

    metrics_df = pd.read_json(bench_path, orient="records")
    for metric_col in ("R2", "R2_log"):
        if metric_col in metrics_df.columns:
            metrics_df = metrics_df[metrics_df[metric_col] >= -1e6]

    metrics_df_plot = metrics_df.copy()
    for metric_col in ("RMSE", "MAE"):
        if metric_col in metrics_df_plot.columns:
            metrics_df_plot[metric_col] = metrics_df_plot[metric_col] * INDEX_RAW_DISPLAY_SCALE

    pred_dir = results_dir / "predictions_regression"
    pred_store: dict = {}
    if pred_dir.exists():
        for csv_path in sorted(pred_dir.glob("*_predictions.csv")):
            ticker = csv_path.stem.replace("_predictions", "")
            pred_store[ticker] = pd.read_csv(
                csv_path,
                index_col="Date",
                parse_dates=True,
            )
    else:
        print(f"WARNING: {pred_dir} not found — skipping prediction plots.")

    sample_tickers = [ticker for ticker in args.tickers if ticker in pred_store]
    if not sample_tickers:
        sample_tickers = sorted(pred_store.keys())

    if pred_store and not args.skip_predictions:
        print(f"\nPlotting {metric_tag}-selected predictions for: {sample_tickers}")
        predictions_dir = plots_dir / f"predictions_{metric_tag}"
        print(f"Writing {metric_tag}-selection plots to {predictions_dir}")
        plot_ticker_predictions(
            pred_store,
            metrics_df,
            sample_tickers,
            save_dir=str(predictions_dir),
            use_log=use_log,
            year_filter=None,
            test_only=True,
            value_display_scale=INDEX_RAW_DISPLAY_SCALE,
            value_label="Realized Variance (Y_fwd, percent-squared units)",
        )

    if not args.skip_summary:
        summary = summarize_benchmarks(
            metrics_df_plot,
            use_log=use_log,
            selection=args.selection,
        )
        plot_summary_metrics(
            summary,
            save_path=str(plots_dir / f"summary_metrics_{metric_tag}.png"),
        )

    if not args.skip_neighbourhoods:
        focus = args.tickers
        graphs_dir = results_dir / "graphs"

        for fname, tag in [
            ("sqcorr_k5.json", "sqcorr"),
            ("pcorr_k5.json", "pcorr"),
            ("mi_k5.json", "mi"),
        ]:
            gpath = graphs_dir / fname
            if not gpath.exists():
                print(f"WARNING: {gpath} not found — skipping {tag} neighbourhood plot.")
                continue
            print(f"\nPlotting neighbourhood change for {tag} (k=5) ...")
            plot_neighbourhood_change(
                graph_json_path=gpath,
                focus_tickers=focus,
                k=5,
                save_path=str(plots_dir / f"neighbourhood_change_{tag}_k5.png"),
            )

    print(f"\nAll plots saved to {plots_dir}")
