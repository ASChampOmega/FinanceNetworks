"""
data/graph_cache.py
====================
Serialize / deserialize the expensive output of
``FinanceNetworkBase.fit_transform()`` — the ``data_dicts`` dictionaries of
DataFrames with ``net_*`` columns — so that classification experiments can
reuse the graph-built features from regression without re-computing them.

The ``net`` objects themselves (with their ``_snapshots`` attribute) are also
cached so that ``save_graph_snapshots`` can be called without re-fitting.

Usage
-----
    from data.graph_cache import save_graph_data, load_graph_data

    # After building:
    save_graph_data(data_dicts_net, nets, cache_dir, "sqcorr")

    # In another script:
    data_dicts_net, nets = load_graph_data(cache_dir, "sqcorr", KNN_VALUES)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _cache_path(cache_dir: Path, tag: str, k: int, kind: str) -> Path:
    """Return the path for a cached object.

    kind is one of "data" (the transformed data_dict) or "net" (the fitted
    network object).
    """
    return cache_dir / f"{tag}_k{k}_{kind}.pkl"


def save_graph_data(
    data_dicts: Dict[int, Dict[str, pd.DataFrame]],
    nets: Dict[int, object],
    cache_dir: Path,
    tag: str,
) -> None:
    """Persist graph-built data_dicts and fitted network objects to disk.

    Parameters
    ----------
    data_dicts : {k_val: {ticker: DataFrame}} — output of fit_transform for
                 each k value.
    nets       : {k_val: fitted FinanceNetworkBase} — the fitted network
                 objects (needed for graph snapshots / interpretability).
    cache_dir  : Directory to write pickle files into.
    tag        : Identifier prefix, e.g. "sqcorr", "pcorr", "exp", "mi".
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for k_val, dd in data_dicts.items():
        with open(_cache_path(cache_dir, tag, k_val, "data"), "wb") as f:
            pickle.dump(dd, f, protocol=pickle.HIGHEST_PROTOCOL)
    for k_val, net in nets.items():
        with open(_cache_path(cache_dir, tag, k_val, "net"), "wb") as f:
            pickle.dump(net, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph_data(
    cache_dir: Path,
    tag: str,
    knn_values: list[int],
) -> tuple[Dict[int, Dict[str, pd.DataFrame]], Dict[int, object]]:
    """Load cached graph-built data_dicts and network objects.

    Returns
    -------
    (data_dicts, nets) — same structure as what was passed to save_graph_data.

    Raises
    ------
    FileNotFoundError if any expected cache file is missing.
    """
    data_dicts: Dict[int, Dict[str, pd.DataFrame]] = {}
    nets: Dict[int, object] = {}
    for k_val in knn_values:
        dp = _cache_path(cache_dir, tag, k_val, "data")
        np_ = _cache_path(cache_dir, tag, k_val, "net")
        if not dp.exists():
            raise FileNotFoundError(f"Graph cache missing: {dp}")
        if not np_.exists():
            raise FileNotFoundError(f"Graph cache missing: {np_}")
        with open(dp, "rb") as f:
            data_dicts[k_val] = pickle.load(f)
        with open(np_, "rb") as f:
            nets[k_val] = pickle.load(f)
    return data_dicts, nets


def graph_cache_exists(
    cache_dir: Path,
    tag: str,
    knn_values: list[int],
) -> bool:
    """Return True if cached data exists for ALL k values under *tag*."""
    for k_val in knn_values:
        if not _cache_path(cache_dir, tag, k_val, "data").exists():
            return False
        if not _cache_path(cache_dir, tag, k_val, "net").exists():
            return False
    return True
