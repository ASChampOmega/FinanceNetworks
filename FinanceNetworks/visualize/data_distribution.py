from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.load_data import get_data
from data.preprocess import preprocess


def _get_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a DatetimeIndex from df['Date'] or df.index."""
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"])
        return pd.DatetimeIndex(dt)
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    # try to coerce index
    return pd.DatetimeIndex(pd.to_datetime(df.index))


def _get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series for column col."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available columns: {list(df.columns)}")
    s = pd.to_numeric(df[col], errors="coerce")
    return s


def _slice_date(
    idx: pd.DatetimeIndex, s: pd.Series, start: Optional[str], end: Optional[str]
) -> Tuple[pd.DatetimeIndex, pd.Series]:
    if start is None and end is None:
        return idx, s
    start_dt = pd.to_datetime(start) if start is not None else None
    end_dt = pd.to_datetime(end) if end is not None else None

    mask = np.ones(len(idx), dtype=bool)
    if start_dt is not None:
        mask &= idx >= start_dt
    if end_dt is not None:
        mask &= idx <= end_dt

    return idx[mask], s.iloc[np.where(mask)[0]]


# ----------------------------
# 1) Price over time
# ----------------------------

def plot_prices_over_time(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    col: str = "Adj Close",
    col_name: str = "Adjusted Close",
    normalize: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    figsize: Tuple[int, int] = (11, 5),
) -> None:
    """
    Plot price series for multiple tickers.
    If normalize=True, each series is scaled to 100 at its first valid point (better for comparison).
    """
    plt.figure(figsize=figsize)
    plotted_any = False

    for t in tickers:
        if t not in data:
            print(f"[warn] Missing ticker in data dict: {t}")
            continue

        df = data[t]
        idx = _get_datetime_index(df)
        s = _get_series(df, col)

        idx, s = _slice_date(idx, s, start, end)
        s = s.dropna()
        if s.empty:
            print(f"[warn] No valid '{col}' data for {t} in the chosen date range.")
            continue

        if normalize:
            base = s.iloc[0]
            if base != 0:
                s = 100.0 * (s / base)

        plt.plot(idx[:len(s)], s.values, label=t)
        plotted_any = True

    if not plotted_any:
        print("[warn] Nothing plotted (no valid series).")
        return

    plt.title(f"{'Normalized ' if normalize else ''}{col_name} over time")
    plt.xlabel("Date")
    plt.ylabel(f"{col_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# 2) Returns over time
# ----------------------------

def plot_returns_over_time(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    col: str = "Returns",
    col_name: str = "Returns",
    start: Optional[str] = None,
    end: Optional[str] = None,
    figsize: Tuple[int, int] = (11, 5),
    subplots: bool = True,
) -> None:
    """
    Plot returns series.
    Default subplots=True because overlays can be visually noisy.
    """
    valid = [t for t in tickers if t in data]
    if not valid:
        print("[warn] No tickers found in data dict.")
        return

    if subplots:
        fig, axes = plt.subplots(len(valid), 1, figsize=(figsize[0], max(figsize[1], 2 * len(valid))), sharex=True)
        if len(valid) == 1:
            axes = [axes]

        for ax, t in zip(axes, valid):
            df = data[t]
            idx = _get_datetime_index(df)
            s = _get_series(df, col)

            idx, s = _slice_date(idx, s, start, end)
            s = s.dropna()
            if s.empty:
                ax.set_title(f"{t} (no valid '{col}' in range)")
                continue

            ax.plot(idx[:len(s)], s.values)
            ax.set_title(t)
            ax.set_ylabel(col_name)

        axes[-1].set_xlabel("Date")
        fig.suptitle(f"{col_name} over time", y=0.995)
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=figsize)
        plotted_any = False
        for t in valid:
            df = data[t]
            idx = _get_datetime_index(df)
            s = _get_series(df, col)

            idx, s = _slice_date(idx, s, start, end)
            s = s.dropna()
            if s.empty:
                continue

            plt.plot(idx[:len(s)], s.values, label=t)
            plotted_any = True

        if not plotted_any:
            print("[warn] Nothing plotted (no valid series).")
            return

        plt.title(f"{col_name} over time")
        plt.xlabel("Date")
        plt.ylabel(col_name)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ----------------------------
# 3) ACF plot (single graph)
# ----------------------------

def plot_acf_column(
    data: Dict[str, pd.DataFrame],
    ticker: str,
    col: str = "Returns",
    col_name: str = "Returns",
    lags: int = 60,
    start: Optional[str] = None,
    end: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 4),
    include_confidence: bool = True,
) -> None:
    """
    Plot ACF for one ticker and one column on a single graph.

    - Uses statsmodels if available; otherwise falls back to a simple autocorrelation computation.
    - Confidence bands are approximate (±1.96/sqrt(N)) if include_confidence=True.
    """
    if ticker not in data:
        raise KeyError(f"Ticker '{ticker}' not found in data dict.")

    df = data[ticker]
    idx = _get_datetime_index(df)
    s = _get_series(df, col)

    idx, s = _slice_date(idx, s, start, end)
    s = s.dropna()
    if s.empty:
        raise ValueError(f"No valid '{col}' data for {ticker} in the chosen date range.")

    x = s.values
    n = len(x)
    max_lag = min(lags, n - 1)
    if max_lag < 1:
        raise ValueError(f"Not enough data points for ACF (n={n}).")

    # Try statsmodels ACF if installed
    acf_vals = None
    try:
        from statsmodels.tsa.stattools import acf as sm_acf
        acf_vals = sm_acf(x, nlags=max_lag, fft=True)
    except Exception:
        # Fallback: compute autocorr for each lag
        x_centered = x - np.mean(x)
        denom = np.sum(x_centered ** 2)
        acf_vals = np.empty(max_lag + 1, dtype=float)
        acf_vals[0] = 1.0
        for k in range(1, max_lag + 1):
            num = np.sum(x_centered[k:] * x_centered[:-k])
            acf_vals[k] = num / denom if denom != 0 else np.nan

    lags_arr = np.arange(len(acf_vals))

    plt.figure(figsize=figsize)
    plt.stem(lags_arr, acf_vals, use_line_collection=True)
    plt.title(f"ACF of {ticker} — {col_name}")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")

    if include_confidence:
        # Approximate 95% confidence bounds for white noise ACF
        conf = 1.96 / np.sqrt(n)
        plt.axhline(conf)
        plt.axhline(-conf)

    plt.tight_layout()
    plt.show()

def plot_acf_many(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    col: str = "Returns",
    col_name: str = "Returns",
    lags: int = 60,
    start: Optional[str] = None,
    end: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
    include_confidence: bool = True,
    confidence_alpha: float = 0.20,
    skip_lag0: bool = True,
    ylim: Union[str, Tuple[float, float]] = "auto",
) -> None:
    """
    Overlay ACF curves for multiple tickers on a single plot.

    - ylim="auto" chooses a symmetric limit based on max |acf| (excluding lag0), with a sane cap.
    - If include_confidence=True, adds +/- 1.96/sqrt(N) bands using the smallest N among tickers.
    """
    # Compute ACF for each ticker
    acfs = {}
    ns = []

    def _acf_values(x: np.ndarray, max_lag: int) -> np.ndarray:
        # Try statsmodels if available, else fallback
        try:
            from statsmodels.tsa.stattools import acf as sm_acf
            return sm_acf(x, nlags=max_lag, fft=True)
        except Exception:
            x_centered = x - np.mean(x)
            denom = np.sum(x_centered ** 2)
            out = np.empty(max_lag + 1, dtype=float)
            out[0] = 1.0
            for k in range(1, max_lag + 1):
                num = np.sum(x_centered[k:] * x_centered[:-k])
                out[k] = num / denom if denom != 0 else np.nan
            return out

    for t in tickers:
        if t not in data:
            print(f"[warn] Missing ticker in data dict: {t}")
            continue

        df = data[t]
        idx = _get_datetime_index(df)
        s = _get_series(df, col)
        idx, s = _slice_date(idx, s, start, end)
        s = s.dropna()
        if len(s) < 5:
            print(f"[warn] Not enough data for ACF: {t} (n={len(s)})")
            continue

        x = s.values
        n = len(x)
        ns.append(n)
        max_lag = min(lags, n - 1)
        if max_lag < 1:
            print(f"[warn] Not enough data for ACF lags: {t} (n={n})")
            continue

        acf_vals = _acf_values(x, max_lag)
        acfs[t] = acf_vals

    if not acfs:
        print("[warn] Nothing plotted (no valid ACF series).")
        return

    # Determine lags array (use the minimum available max lag across tickers so curves align)
    min_len = min(len(v) for v in acfs.values())
    lags_arr = np.arange(min_len)

    # Build auto y-limits
    if ylim == "auto":
        max_abs = 0.0
        for v in acfs.values():
            vv = v[:min_len]
            if skip_lag0 and len(vv) > 1:
                vv = vv[1:]
            vv = vv[np.isfinite(vv)]
            if vv.size:
                max_abs = max(max_abs, float(np.max(np.abs(vv))))
        # Sane defaults: at least 0.10, cap at 0.60, add headroom
        max_abs = min(max(0.10, 1.15 * max_abs), 0.60)
        ylim_used = (-max_abs, max_abs)
    else:
        ylim_used = ylim  # type: ignore

    # Confidence band based on smallest N (most conservative)
    conf = None
    if include_confidence and ns:
        n_min = min(ns)
        conf = 1.96 / np.sqrt(n_min)

    # Plot
    plt.figure(figsize=figsize)
    for t, v in acfs.items():
        vv = v[:min_len]
        if skip_lag0:
            plt.plot(lags_arr[1:], vv[1:], label=t)
        else:
            plt.plot(lags_arr, vv, label=t)

    if include_confidence and conf is not None:
        x_band = lags_arr[1:] if skip_lag0 else lags_arr
        plt.fill_between(
            x_band, conf, -conf, alpha=confidence_alpha, linewidth=0
        )
        plt.axhline(conf, linewidth=1)
        plt.axhline(-conf, linewidth=1)

    plt.axhline(0.0, linewidth=1)
    plt.title(f"ACF overlay — {col_name}")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.ylim(ylim_used)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

from typing import Literal

def plot_returns_distribution(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    col: str = "Returns",
    col_name: str = "Returns",
    start: Optional[str] = None,
    end: Optional[str] = None,
    bins: int = 80,
    mode: Literal["overlay", "subplots"] = "overlay",
    density: bool = True,
    clip_quantiles: Tuple[float, float] = (0.005, 0.995),
    figsize: Tuple[int, int] = (10, 4),
    show_normal_fit: bool = False,
):
    """
    Plot return distributions for one or more tickers.

    - mode="overlay": all tickers on one axis using step histograms (cleaner).
    - mode="subplots": one histogram per ticker.
    - clip_quantiles keeps x-limits reasonable by ignoring extreme tails.
    - show_normal_fit is best used when tickers has length 1.
    Returns a DataFrame of summary stats.
    """
    series_map = {}
    for t in tickers:
        if t not in data:
            print(f"[warn] Missing ticker in data dict: {t}")
            continue
        df = data[t]
        idx = _get_datetime_index(df)
        s = _get_series(df, col)
        idx, s = _slice_date(idx, s, start, end)
        s = s.dropna()
        if s.empty:
            print(f"[warn] No valid '{col}' data for {t}")
            continue
        series_map[t] = s

    if not series_map:
        print("[warn] Nothing plotted (no valid return series).")
        return pd.DataFrame()

    # Determine a shared x-range from pooled quantiles (nice-looking plots)
    pooled = pd.concat(series_map.values(), axis=0).dropna()
    lo, hi = pooled.quantile(clip_quantiles[0]), pooled.quantile(clip_quantiles[1])

    # Summary stats (handy for your report)
    stats = []
    for t, s in series_map.items():
        ss = s[(s >= lo) & (s <= hi)]
        stats.append({
            "Ticker": t,
            "N": int(len(s)),
            "Mean": float(s.mean()),
            "Std": float(s.std(ddof=1)),
            "Skew": float(s.skew()),
            "Kurtosis": float(s.kurtosis()),  # excess kurtosis
        })
    stats_df = pd.DataFrame(stats).set_index("Ticker")

    if mode == "overlay":
        plt.figure(figsize=figsize)
        for t, s in series_map.items():
            ss = s[(s >= lo) & (s <= hi)]
            plt.hist(
                ss.values,
                bins=bins,
                density=density,
                histtype="step",
                linewidth=1.5,
                label=t
            )

        # Optional normal fit (recommended only for single ticker)
        if show_normal_fit and len(series_map) == 1:
            t = next(iter(series_map.keys()))
            mu, sd = stats_df.loc[t, "Mean"], stats_df.loc[t, "Std"]
            xs = np.linspace(lo, hi, 400)
            pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
            plt.plot(xs, pdf, linewidth=2, label=f"{t} Normal fit")

        plt.title(f"Distribution of {col_name} (clipped to {clip_quantiles[0]:.3f}-{clip_quantiles[1]:.3f} quantiles)")
        plt.xlabel(col_name)
        plt.ylabel("Density" if density else "Count")
        plt.xlim((lo, hi))
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.show()

    else:
        n = len(series_map)
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], max(figsize[1], 2*n)), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, (t, s) in zip(axes, series_map.items()):
            ss = s[(s >= lo) & (s <= hi)]
            ax.hist(ss.values, bins=bins, density=density)
            ax.set_title(t)
            ax.set_ylabel("Density" if density else "Count")

        axes[-1].set_xlabel(col_name)
        axes[-1].set_xlim((lo, hi))
        fig.suptitle(f"Distribution of {col_name} (clipped)", y=0.995)
        plt.tight_layout()
        plt.show()

    return stats_df

def plot_return_outliers(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    col: str = "Returns",
    col_name: str = "Returns",
    start: Optional[str] = None,
    end: Optional[str] = None,
    method: Literal["quantile", "sigma"] = "quantile",
    # quantile method params:
    tail_q: float = 0.005,  # shows bottom tail_q and top tail_q
    # sigma method params:
    z: float = 3.0,
    bins: int = 60,
    mode: Literal["overlay", "subplots"] = "overlay",
    density: bool = True,
    figsize: Tuple[int, int] = (10, 4),
    show_cutoffs: bool = True,
) -> pd.DataFrame:
    """
    Plot only outliers (tails) of return distributions for given tickers.

    Outlier definition:
      - method="quantile": outliers are r <= q_low or r >= q_high,
        where q_low=tail_q and q_high=1-tail_q quantiles (per ticker).
      - method="sigma": outliers are r <= mean - z*std or r >= mean + z*std (per ticker).

    Returns a summary DataFrame with thresholds and outlier counts.
    """

    def _get_outliers(s: pd.Series) -> Tuple[pd.Series, float, float]:
        s = s.dropna()
        if s.empty:
            return s, np.nan, np.nan

        if method == "quantile":
            if not (0.0 < tail_q < 0.5):
                raise ValueError("tail_q must be in (0, 0.5). Example: 0.005 for 0.5% tails.")
            lo = float(s.quantile(tail_q))
            hi = float(s.quantile(1.0 - tail_q))
        elif method == "sigma":
            mu = float(s.mean())
            sd = float(s.std(ddof=1))
            lo = mu - z * sd
            hi = mu + z * sd
        else:
            raise ValueError("method must be 'quantile' or 'sigma'.")

        out = s[(s <= lo) | (s >= hi)]
        return out, lo, hi

    # Collect outliers per ticker + stats
    out_map: Dict[str, pd.Series] = {}
    stats_rows = []

    for t in tickers:
        if t not in data:
            print(f"[warn] Missing ticker in data dict: {t}")
            continue

        df = data[t]
        idx = _get_datetime_index(df)
        s = _get_series(df, col)
        idx, s = _slice_date(idx, s, start, end)
        s = s.dropna()
        if s.empty:
            print(f"[warn] No valid '{col}' data for {t}")
            continue

        out, lo, hi = _get_outliers(s)
        out_map[t] = out

        stats_rows.append({
            "Ticker": t,
            "N": int(len(s)),
            "Outliers": int(len(out)),
            "OutlierFrac": float(len(out) / len(s)) if len(s) > 0 else np.nan,
            "LowerCutoff": lo,
            "UpperCutoff": hi,
        })

    summary = pd.DataFrame(stats_rows).set_index("Ticker") if stats_rows else pd.DataFrame()

    if not out_map:
        print("[warn] Nothing plotted (no outlier series).")
        return summary

    # Pick a global x-range for nicer visuals (based on pooled outliers)
    pooled_out = pd.concat(out_map.values(), axis=0).dropna()
    if pooled_out.empty:
        print("[warn] No outliers found under the chosen definition.")
        return summary

    # Use robust x-lims from outlier quantiles to avoid a single extreme point destroying the plot
    x_lo = float(pooled_out.quantile(0.01))
    x_hi = float(pooled_out.quantile(0.99))
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_lo >= x_hi:
        # fallback
        x_lo, x_hi = float(pooled_out.min()), float(pooled_out.max())

    title_tail = "quantile tails" if method == "quantile" else f"{z:.1f}σ tails"
    title = f"Outlier distribution of {col_name} ({title_tail})"

    if mode == "overlay":
        plt.figure(figsize=figsize)
        for t, out in out_map.items():
            out = out.dropna()
            if out.empty:
                continue
            plt.hist(
                out.values,
                bins=bins,
                density=density,
                histtype="step",
                linewidth=1.5,
                label=t
            )

            if show_cutoffs and t in summary.index and np.isfinite(summary.loc[t, "LowerCutoff"]):
                plt.axvline(summary.loc[t, "LowerCutoff"], linewidth=1)
                plt.axvline(summary.loc[t, "UpperCutoff"], linewidth=1)

        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Density" if density else "Count")
        plt.xlim((x_lo, x_hi))
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.show()

    else:  # subplots
        n = len(out_map)
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], max(figsize[1], 2*n)), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, (t, out) in zip(axes, out_map.items()):
            out = out.dropna()
            ax.hist(out.values, bins=bins, density=density)
            ax.set_title(t)
            ax.set_ylabel("Density" if density else "Count")

            if show_cutoffs and t in summary.index and np.isfinite(summary.loc[t, "LowerCutoff"]):
                ax.axvline(summary.loc[t, "LowerCutoff"], linewidth=1)
                ax.axvline(summary.loc[t, "UpperCutoff"], linewidth=1)

        axes[-1].set_xlabel(col)
        axes[-1].set_xlim((x_lo, x_hi))
        fig.suptitle(title, y=0.995)
        plt.tight_layout()
        plt.show()

    return summary

if __name__ == "__main__":
    data = get_data()
    data = preprocess(data)
    plot_prices_over_time(
        data,
        ['AAPL', 'GOOG', 'META', 'MSFT', 'TSLA']
    )
    plot_prices_over_time(
        data,
        ['AAPL', 'GOOG', 'META', 'MSFT', 'TSLA'],
        col="Y_fwd",
        col_name = "Weekly Volatility",
    )
    plot_acf_many(
        data,
        ['AAPL', 'GOOG', 'META', 'MSFT', 'TSLA'],
        col="Returns",
        col_name = "Weekly Volatility",
        lags=60
    )
    
    plot_returns_distribution(
        data,
        ['AAPL', 'GOOG', 'META', 'MSFT', 'TSLA'],
        col="Y_fwd",
        col_name = "Weekly Volatility",
        mode='subplots'
    )
    
    plot_return_outliers(
        data,
        ['AAPL', 'GOOG', 'META', 'MSFT', 'TSLA'],
        col="Y_fwd",
        col_name = "Weekly Volatility",
        mode='subplots'
    )