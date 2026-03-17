"""
evaluation/interpretability.py
===============================
Utilities for extracting, serializing, and persisting fitted model parameters
and network graph snapshots for post-hoc interpretability analysis.

Two main capabilities:
1. **Model parameter extraction** -- given a fitted model instance (any of the
   HAR / ARIMA / GARCH / Regime-Switching / Network variants, for
   both regression and classification), produce a JSON-serializable dictionary
   of all learned parameters (coefficients, intercepts, scaler statistics,
   regime thresholds, ARIMA/GARCH params, etc.).

2. **Graph snapshot serialization** -- given a fitted FinanceNetworkBase
   instance, serialize every rolling-window graph snapshot (nodes, edges with
   weights, graph-level statistics) to a JSON-friendly list of dicts.

Usage
-----
These utilities are called automatically inside ``run_benchmarks_multi_fold``
(cross_val.py) and ``run_classification_cv`` (classification.py) when the
``save_params=True`` flag is set.  They can also be used standalone::

    from evaluation.interpretability import extract_model_params, save_graph_snapshots
    params = extract_model_params(fitted_model)
    save_graph_snapshots(net_obj, results_dir / "graphs")
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Numpy / Pandas serialization helper
# ---------------------------------------------------------------------------

def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.Index):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# Pipeline parameter extraction helpers
# ---------------------------------------------------------------------------

def _extract_pipeline_params(pipe) -> Dict[str, Any]:
    """
    Extract parameters from a sklearn Pipeline containing a StandardScaler
    and a final estimator (LinearRegression, Ridge, Lasso, LogisticRegression).

    Returns a dict with scaler means/scales and estimator coefficients/intercept.
    """
    result: Dict[str, Any] = {}

    # Scaler (named "scaler" by convention in this project)
    if hasattr(pipe, "named_steps") and "scaler" in pipe.named_steps:
        scaler = pipe.named_steps["scaler"]
        if hasattr(scaler, "mean_") and scaler.mean_ is not None:
            result["scaler_mean"] = scaler.mean_.tolist()
            result["scaler_scale"] = scaler.scale_.tolist()

    # Final estimator -- try both "reg" and "clf" naming conventions
    for step_name in ("reg", "clf"):
        if hasattr(pipe, "named_steps") and step_name in pipe.named_steps:
            est = pipe.named_steps[step_name]
            result["estimator_type"] = type(est).__name__
            if hasattr(est, "coef_"):
                coef = est.coef_
                # LogisticRegression stores coef_ as 2-D (n_classes, n_features)
                if coef.ndim == 2 and coef.shape[0] == 1:
                    coef = coef.ravel()
                result["coefficients"] = coef.tolist()
            if hasattr(est, "intercept_"):
                intercept = est.intercept_
                if isinstance(intercept, np.ndarray):
                    intercept = intercept.tolist()
                    if len(intercept) == 1:
                        intercept = intercept[0]
                result["intercept"] = float(intercept) if np.isscalar(intercept) else intercept
            if hasattr(est, "alpha"):
                result["alpha"] = float(est.alpha)
            if hasattr(est, "C"):
                result["C"] = float(est.C)
            break

    return result


# ---------------------------------------------------------------------------
# Model parameter extraction (dispatcher)
# ---------------------------------------------------------------------------

def extract_model_params(model: Any) -> Dict[str, Any]:
    """
    Extract all learned parameters from a fitted model instance.

    Supports every model class in baselines.py, baselines_classification.py,
    network_models.py, and network_models_classification.py.

    Returns
    -------
    A JSON-serializable dictionary.  The structure varies by model type but
    always includes:
      - "model_class": the class name
      - "features": the feature list used by the model

    For Pipeline-based models (HAR, Network* variants):
      - "coefficients", "intercept", "scaler_mean", "scaler_scale"

    For ARIMA:
      - "arima_order", "arima_params" (dict of param name -> value)

    For GARCH:
      - "garch_params" (omega, alpha[], beta[], etc.)

    For Regime-Switching models:
      - "regime_threshold", "fallback_params", "regime_0_params", "regime_1_params"

    For two-stage models (NetworkVAR*):
      - "stage1_params", "stage2_params"
    """
    info: Dict[str, Any] = {
        "model_class": type(model).__name__,
    }
    if hasattr(model, "features"):
        info["features"] = list(model.features)

    class_name = type(model).__name__

    # ── HAR / HARExtended (regression) ────────────────────────────────
    if class_name in ("HARLogRegressor", "HARExtendedLogRegressor"):
        if hasattr(model, "model_") and model.model_ is not None:
            info.update(_extract_pipeline_params(model.model_))

    # ── HAR Logit / HARExtended Logit (classification) ────────────────
    elif class_name in ("HARLogitClassifier", "HARExtendedLogitClassifier"):
        if hasattr(model, "model_") and model.model_ is not None:
            info.update(_extract_pipeline_params(model.model_))

    # ── ARIMA ──────────────────────────────────────────────────────────
    elif class_name == "ARIMALogY":
        info["arima_order"] = list(model.order)
        if hasattr(model, "res_") and model.res_ is not None:
            info["arima_params"] = {
                str(k): float(v) for k, v in model.res_.params.items()
            }
            if hasattr(model.res_, "aic"):
                info["aic"] = float(model.res_.aic)
            if hasattr(model.res_, "bic"):
                info["bic"] = float(model.res_.bic)

    # ── GARCH ──────────────────────────────────────────────────────────
    elif class_name == "GARCHWeeklyRV":
        info["p"] = model.p
        info["q"] = model.q
        info["horizon"] = model.horizon
        info["scale"] = model.scale
        if hasattr(model, "res_") and model.res_ is not None:
            info["garch_params"] = {
                str(k): float(v) for k, v in model.res_.params.items()
            }
            if hasattr(model.res_, "aic"):
                info["aic"] = float(model.res_.aic)
            if hasattr(model.res_, "bic"):
                info["bic"] = float(model.res_.bic)
            if hasattr(model.res_, "loglikelihood"):
                info["loglikelihood"] = float(model.res_.loglikelihood)

    # ── DCC-GARCH (forecasting + classification wrapper) ──────────────
    elif class_name in ("DCCGARCHWeeklyRV", "DCCGARCHSpikeClassifier"):
        m = model
        if class_name == "DCCGARCHSpikeClassifier" and getattr(model, "dcc_", None) is not None:
            m = model.dcc_

        info["p"] = getattr(m, "p", None)
        info["q"] = getattr(m, "q", None)
        info["horizon"] = getattr(m, "horizon", None)
        info["scale"] = getattr(m, "scale", None)
        info["rho_weight"] = getattr(m, "rho_weight", None)
        info["aux_returns_col"] = getattr(m, "aux_returns_col", None)

        if getattr(m, "_dcc_ab", None) is not None:
            info["dcc_alpha"] = float(m._dcc_ab[0])
            info["dcc_beta"] = float(m._dcc_ab[1])

        if getattr(m, "res1_", None) is not None:
            info["garch1_params"] = {
                str(k): float(v) for k, v in m.res1_.params.items()
            }
        if getattr(m, "res2_", None) is not None:
            info["garch2_params"] = {
                str(k): float(v) for k, v in m.res2_.params.items()
            }

        if class_name == "DCCGARCHSpikeClassifier" and getattr(model, "calibrator_", None) is not None:
            info["calibrator_params"] = _extract_pipeline_params(model.calibrator_)

    # ── Regime-Switching HAR (regression) ──────────────────────────────
    elif class_name == "RegimeSwitchingHARLogRegressor":
        info["regime_col"] = model.regime_col
        info["regime_percentile"] = model.regime_percentile
        info["regime_threshold"] = float(model.threshold_)
        if model.fallback_model_ is not None:
            info["fallback_params"] = _extract_pipeline_params(model.fallback_model_)
        for regime_id in (0, 1):
            key = f"regime_{regime_id}_params"
            if regime_id in model.models_:
                info[key] = _extract_pipeline_params(model.models_[regime_id])
            else:
                info[key] = "fallback"

    # ── Regime-Switching HAR (classification) ──────────────────────────
    elif class_name == "RegimeSwitchingHARLogitClassifier":
        info["regime_col"] = model.regime_col
        info["regime_percentile"] = model.regime_percentile
        info["regime_threshold"] = float(model.threshold_)
        if model.fallback_model_ is not None:
            info["fallback_params"] = _extract_pipeline_params(model.fallback_model_)
        for regime_id in (0, 1):
            key = f"regime_{regime_id}_params"
            if regime_id in model.models_:
                info[key] = _extract_pipeline_params(model.models_[regime_id])
            else:
                info[key] = "fallback"

    # ── NetworkHARRegressor (single-stage) ─────────────────────────────
    elif class_name == "NetworkHARRegressor":
        if hasattr(model, "_pipe") and model._pipe is not None:
            info.update(_extract_pipeline_params(model._pipe))

    # ── NetworkVARRegressor (two-stage) ────────────────────────────────
    elif class_name == "NetworkVARRegressor":
        info["stage2_alpha"] = model.stage2_alpha
        info["correction_bound"] = model.correction_bound
        if hasattr(model, "_stage1") and model._stage1 is not None:
            info["stage1_params"] = _extract_pipeline_params(model._stage1)
        if hasattr(model, "_stage2") and model._stage2 is not None:
            info["stage2_params"] = _extract_pipeline_params(model._stage2)

    # ── NetworkHARClassifier (single-stage) ────────────────────────────
    elif class_name == "NetworkHARClassifier":
        if hasattr(model, "_pipe") and model._pipe is not None:
            info.update(_extract_pipeline_params(model._pipe))

    # ── NetworkVARClassifier (two-stage) ───────────────────────────────
    elif class_name == "NetworkVARClassifier":
        info["C_stage1"] = getattr(model, "C_stage1", None)
        info["stage2_alpha"] = model.stage2_alpha
        info["correction_bound"] = model.correction_bound
        if hasattr(model, "_stage1") and model._stage1 is not None:
            info["stage1_params"] = _extract_pipeline_params(model._stage1)
        if hasattr(model, "_stage2") and model._stage2 is not None:
            info["stage2_params"] = _extract_pipeline_params(model._stage2)
    # ── LearnedWeightNetworkHARRegressor ───────────────────────────
    elif class_name == "LearnedWeightNetworkHARRegressor":
        info["k"] = model.k
        info["m"] = model.m
        info["alpha"] = model.alpha
        info["lasso_alpha"] = model.lasso_alpha
        info["use_clustering"] = model.use_clustering
        if model._W is not None:
            info["W"] = model._W.tolist()
        if hasattr(model, "_pipe") and model._pipe is not None:
            info.update(_extract_pipeline_params(model._pipe))

    # ── LearnedWeightNetworkHARClassifier ──────────────────────────
    elif class_name == "LearnedWeightNetworkHARClassifier":
        info["k"] = model.k
        info["m"] = model.m
        info["C"] = model.C
        info["use_clustering"] = model.use_clustering
        if model._W is not None:
            info["W"] = model._W.tolist()
        if hasattr(model, "_pipe") and model._pipe is not None:
            info.update(_extract_pipeline_params(model._pipe))
    # ── Fallback: try common patterns ──────────────────────────────────
    else:
        if hasattr(model, "model_") and model.model_ is not None:
            try:
                info.update(_extract_pipeline_params(model.model_))
            except Exception:
                pass
        if hasattr(model, "_pipe") and model._pipe is not None:
            try:
                info.update(_extract_pipeline_params(model._pipe))
            except Exception:
                pass

    return _to_serializable(info)


# ---------------------------------------------------------------------------
# Graph snapshot serialization
# ---------------------------------------------------------------------------

def serialize_graph_snapshots(net_obj: Any) -> Dict[str, Any]:
    """
    Serialize all rolling-window graph snapshots from a fitted
    FinanceNetworkBase instance to a JSON-friendly dictionary.

    Structure
    ---------
    {
      "network_class": "SquaredCorrelationNetwork",
      "hyperparams": {
        "window": 60, "step": 5, "graph_type": "knn", "k": 5, ...
      },
      "n_snapshots": 150,
      "snapshots": [
        {
          "date": "2010-06-15T00:00:00",
          "n_nodes": 95,
          "n_edges": 237,
          "avg_abs_corr": 0.35,
          "edges": [
            {"source": "AAPL", "target": "MSFT", "weight": 0.152},
            ...
          ]
        },
        ...
      ]
    }

    Nodes are stored implicitly as the union of all edge endpoints to save
    space.  The full node list for each snapshot can be recovered from the
    number of nodes. Isolated nodes (degree 0) are included via the
    "nodes" field.
    """
    result: Dict[str, Any] = {
        "network_class": type(net_obj).__name__,
        "hyperparams": {
            "window": net_obj.window,
            "step": net_obj.step,
            "graph_type": net_obj.graph_type,
            "threshold": net_obj.threshold,
            "k": net_obj.k,
            "feature_cols": list(net_obj.feature_cols),
            "returns_col": net_obj.returns_col,
            "min_obs_frac": net_obj.min_obs_frac,
            "min_tickers": net_obj.min_tickers,
            "idw_kernel": net_obj.idw_kernel,
        },
        "tickers": list(net_obj.tickers_),
        "n_snapshots": len(net_obj._snapshots),
    }

    # Optional: shrinkage for PartialCorrelationNetwork
    if hasattr(net_obj, "shrinkage"):
        result["hyperparams"]["shrinkage"] = net_obj.shrinkage
    if hasattr(net_obj, "n_bins"):
        result["hyperparams"]["n_bins"] = net_obj.n_bins
    if hasattr(net_obj, "exp_lambda"):
        result["hyperparams"]["exp_lambda"] = net_obj.exp_lambda

    snapshots_list: List[Dict[str, Any]] = []
    avg_corr_map = getattr(net_obj, "_snap_avg_abs_corr", {})

    for date, G in net_obj._snapshots:
        snap: Dict[str, Any] = {
            "date": date.isoformat(),
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "avg_abs_corr": float(avg_corr_map.get(date, 0.0)),
        }

        # All nodes (including isolates)
        snap["nodes"] = sorted(G.nodes())

        # Edges with weights
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": round(float(data.get("weight", 1.0)), 6),
            })
        snap["edges"] = edges

        snapshots_list.append(snap)

    result["snapshots"] = snapshots_list
    return _to_serializable(result)


def serialize_feature_snapshots(
    data_dict: Dict[str, pd.DataFrame],
    tickers: Optional[List[str]] = None,
    prefix: str = "net_",
) -> Dict[str, Any]:
    """
    Serialize per-ticker network feature time series to a JSON-friendly dict.

    Parameters
    ----------
    data_dict : Dict[str, DataFrame]
        Per-ticker feature DataFrames (typically output of fit_transform).
    tickers : Optional[List[str]]
        If provided, only serialize these tickers that exist in data_dict.
    prefix : str
        Column prefix used to select network features (default: "net_").
    """
    selected = tickers or sorted(data_dict.keys())
    out: Dict[str, Any] = {
        "n_tickers": 0,
        "feature_prefix": prefix,
        "tickers": {},
    }

    for t in selected:
        if t not in data_dict:
            continue
        df = data_dict[t]
        net_cols = [c for c in df.columns if c.startswith(prefix)]
        if not net_cols:
            continue

        snap_df = df[net_cols].copy()
        snap_df.index = pd.to_datetime(snap_df.index)
        snap_df = snap_df.replace([np.inf, -np.inf], np.nan)
        snap_df = snap_df.reset_index().rename(columns={"index": "Date"})
        snap_df["Date"] = snap_df["Date"].dt.strftime("%Y-%m-%d")

        out["tickers"][t] = {
            "n_rows": int(len(snap_df)),
            "columns": net_cols,
            "rows": _to_serializable(snap_df.to_dict(orient="records")),
        }

    out["n_tickers"] = len(out["tickers"])
    return _to_serializable(out)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model_params(
    params_store: List[Dict[str, Any]],
    results_dir: Path,
    filename: str = "model_params.json",
) -> Path:
    """
    Persist extracted model parameters to a JSON file.

    Parameters
    ----------
    params_store : List of dicts, each produced by extract_model_params()
                   plus metadata (ticker, fold, category, model_name,
                   train window, test window).
    results_dir  : Directory to write the file into (created if needed).
    filename     : Output filename.

    Returns
    -------
    Path to the written file.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    with open(out_path, "w") as f:
        json.dump(_to_serializable(params_store), f, indent=2, default=str)

    print(f"Model parameters    -> {out_path}  ({len(params_store)} records)")
    return out_path


def save_feature_snapshots(
    data_dict: Dict[str, pd.DataFrame],
    results_dir: Path,
    name: str = "feature_snapshots",
    tickers: Optional[List[str]] = None,
    prefix: str = "net_",
) -> Path:
    """
    Serialize and persist per-ticker feature snapshots for interpretability.

    Parameters
    ----------
    data_dict  : Per-ticker feature DataFrames.
    results_dir: Directory to write into (created if needed).
    name       : Base filename (without extension).
    tickers    : Optional subset of tickers to include.
    prefix     : Column prefix used to filter feature columns.

    Returns
    -------
    Path to the written JSON file.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"

    data = serialize_feature_snapshots(data_dict, tickers=tickers, prefix=prefix)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Feature snapshots   -> {out_path}  ({data['n_tickers']} tickers)")
    return out_path


def save_graph_snapshots(
    net_obj: Any,
    results_dir: Path,
    name: str = "graph_snapshots",
) -> Path:
    """
    Serialize and persist all graph snapshots from a fitted network object.

    Parameters
    ----------
    net_obj     : A fitted FinanceNetworkBase subclass instance.
    results_dir : Directory to write into (created if needed).
    name        : Base filename (without extension).

    Returns
    -------
    Path to the written JSON file.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"

    data = serialize_graph_snapshots(net_obj)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    n_snaps = data["n_snapshots"]
    print(f"Graph snapshots     -> {out_path}  ({n_snaps} snapshots)")
    return out_path
