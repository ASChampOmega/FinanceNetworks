"""
models/network_models_classification.py
========================================
Network-augmented volatility-spike classifiers.

Mirrors ``network_models.py`` exactly in structure but replaces the
regression objective with logistic regression.  Both models expect that
the per-ticker DataFrames have been enriched with ``net_*`` columns by
``FinanceNetworkBase.fit_transform()`` before cross-validation.

Models
------
NetworkHARClassifier
    Single-stage L2-penalised logistic regression on the full HAR-Extended
    feature set plus network structural features (degree, neighbourhood
    turnover, IDW RVs of neighbours).  Mirrors ``NetworkHARRegressor``.

NetworkVARClassifier
    Two-stage network error-correction classifier.  Mirrors
    ``NetworkVARRegressor``:

      Stage 1 — Logistic HAR on own features only.
                decision_function(X) gives the log-odds of a spike.

      Stage 2 — Ridge regression on network features, fitted on the
                probability residuals (y_true − p1_train) from stage 1.
                The correction is clipped to [-bound, +bound] in log-odds
                space to prevent explosive outputs.

      Final probability — sigmoid(log_odds_1 + clip(correction))

    Motivation: the HAR logit residuals (unexplained spike probability)
    contain cross-sectional contagion that own-stock lags cannot capture.
    The network features absorb this residual component in a second stage,
    using independent regularisation so network corrections stay bounded.

Network features expected (produced by FinanceNetworkBase.transform())
----------------------------------------------------------------------
    net_degree            node degree in the current graph snapshot
    net_degree_change     neighbourhood turnover (0 = stable, 1 = full rewire)
    net_idw_log_RV1       IDW mean of neighbours' log_RV1
    net_idw_log_RV5       IDW mean of neighbours' log_RV5
    net_idw_log_RV22      IDW mean of neighbours' log_RV22

    Optionally (when use_clustering=True):
    net_node_clustering   local clustering coefficient
    net_global_clustering market-wide clustering coefficient
    net_avg_abs_corr      mean absolute correlation among all pairs

Missing net_* values are imputed with 0.0 (same as the regression models).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Re-use the shared feature lists and helpers from network_models.py
from models.network_models import (
    HAR_FEATURES,
    NET_FEATURES,
    CLUSTERING_FEATURES,
    NET_FEATURES_FULL,
    _fill_net,
    _select_net_features,
    _dedupe_preserve_order,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _make_logit(C: float, max_iter: int) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
    )


# ---------------------------------------------------------------------------
# NetworkHARClassifier  (single-stage)
# ---------------------------------------------------------------------------

class NetworkHARClassifier(BaseEstimator, ClassifierMixin):
    """
    Single-stage logistic regression on HAR-Extended + network features.

    Model
    -----
    P(spike | log_RV1..log_pos_semi5, Returns,
              net_degree, net_degree_change,
              net_idw_log_RV1..net_idw_log_RV22 [, clustering features])

    ``Returns`` supplies the leverage effect: negative return days tend to
    raise the probability of a future volatility spike disproportionately
    relative to same-magnitude positive return days.

    Parameters
    ----------
    C             : Inverse regularisation strength (L2 logistic).
    max_iter      : Solver iteration cap.
    use_clustering: If True, add clustering features (net_node_clustering,
                    net_global_clustering, net_avg_abs_corr).
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1_000,
        use_clustering: bool = False,
    ):
        self.C = C
        self.max_iter = max_iter
        self.use_clustering = use_clustering
        net_feats = _select_net_features(use_clustering)
        self.features: List[str] = HAR_FEATURES + net_feats
        self.model_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NetworkHARClassifier":
        X_fit = _fill_net(X)[self.features]
        self.model_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", _make_logit(self.C, self.max_iter)),
            ]
        )
        self.model_.fit(X_fit, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(_fill_net(X)[self.features])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(_fill_net(X)[self.features])


# ---------------------------------------------------------------------------
# NetworkVARClassifier  (two-stage error-correction)
# ---------------------------------------------------------------------------

class NetworkVARClassifier(BaseEstimator, ClassifierMixin):
    """
    Two-stage network error-correction classifier.

    Motivation
    ----------
    A spike predicted by the HAR logit using only own-stock RV lags will
    carry systematic residual probability that correlates with the network
    neighbourhood.  The second stage learns that residual signal from
    network features under a separate ridge penalty, then adds it (in
    log-odds space) to the HAR stage-1 output.

    Stage 1 — HAR Logistic (own features)
        Fit logistic regression on HAR_FEATURES (including Returns).
        Store training log-odds: d1 = decision_function(X_train)
        Store training predicted probability: p1 = sigmoid(d1)
        Compute probability residuals: r = y_train − p1

    Stage 2 — Ridge correction on network features
        Fit Ridge(alpha=stage2_alpha) with target = r and
        predictors = net_* features.
        The correction is bounded to [-correction_bound, +correction_bound]
        in log-odds space.

    Prediction
        log_odds_final = d1 + clip(correction, -bound, +bound)
        P(spike) = sigmoid(log_odds_final)
        hard label: 1 if P(spike) >= 0.5

    Parameters
    ----------
    C_stage1         : Inverse L2 strength for the HAR logit (stage 1).
    stage2_alpha     : Ridge alpha for the network correction (stage 2).
    correction_bound : Maximum absolute log-odds shift from stage 2.
    use_clustering   : Include clustering network features.
    max_iter         : Iteration cap for the stage-1 logistic fit.
    """

    def __init__(
        self,
        C_stage1: float = 1.0,
        stage2_alpha: float = 1.0,
        correction_bound: float = 2.0,
        use_clustering: bool = False,
        max_iter: int = 1_000,
    ):
        self.C_stage1 = C_stage1
        self.stage2_alpha = stage2_alpha
        self.correction_bound = correction_bound
        self.use_clustering = use_clustering
        self.max_iter = max_iter
        self._net_feats: List[str] = _select_net_features(use_clustering)
        self.features: List[str] = HAR_FEATURES + self._net_feats
        self._stage1: Optional[Pipeline] = None
        self._stage2: Optional[Pipeline] = None

    # ── Stage 1: own HAR logit ────────────────────────────────────────────

    def _fit_stage1(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Fit stage-1 logit; return training probability residuals."""
        self._stage1 = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", _make_logit(self.C_stage1, self.max_iter)),
            ]
        )
        self._stage1.fit(X[HAR_FEATURES], y)
        # decision_function gives shape (n,) log-odds for binary classifiers
        log_odds = self._stage1.decision_function(X[HAR_FEATURES])
        p1 = _sigmoid(log_odds)
        return y.values.astype(float) - p1  # probability residuals

    # ── Stage 2: network ridge correction ────────────────────────────────

    def _fit_stage2(self, X: pd.DataFrame, residuals: np.ndarray) -> None:
        """Fit stage-2 Ridge on network features with prob-residuals as target."""
        X_net = _fill_net(X)[self._net_feats]
        self._stage2 = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=self.stage2_alpha)),
            ]
        )
        self._stage2.fit(X_net, residuals)

    # ── sklearn API ───────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NetworkVARClassifier":
        residuals = self._fit_stage1(X, y)
        self._fit_stage2(X, residuals)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        log_odds1 = self._stage1.decision_function(X[HAR_FEATURES])
        correction = self._stage2.predict(_fill_net(X)[self._net_feats])
        correction = np.clip(correction, -self.correction_bound, self.correction_bound)
        p_spike = _sigmoid(log_odds1 + correction)
        return np.column_stack([1.0 - p_spike, p_spike])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
