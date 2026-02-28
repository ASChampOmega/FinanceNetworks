"""
network_models.py
=================
Network-augmented volatility models.

All models in this file expect that the per-ticker DataFrames have already
been enriched with net_* columns by ``FinanceNetworkBase.fit_transform()``
before cross-validation begins.  See ``models/correlation_network.py`` for
the offline build pattern.

Models
------
NetworkHARRegressor
    Single-stage Lasso regression using the full HAR-Extended feature set
    plus network structural features (degree, neighbourhood turnover, IDW RVs).
    Straightforward baseline for network-augmented prediction.

NetworkVARRegressor
    Two-stage "network error correction" model:
      Stage 1 -- Standard HAR-Extended OLS fit in log space (own features only).
      Stage 2 -- Ridge regression of the stage-1 log-space residuals on the
                 network features (degree, turnover, IDW RVs of neighbours).
      Prediction -- exp(stage1_log_pred + stage2_log_correction).

    Motivation: the HAR residuals contain common-factor / contagion variance
    that the own-stock lags cannot explain.  The network features capture this
    cross-sectional component.  Fitting the two stages separately prevents the
    network terms from being obscured by the much larger HAR coefficients and
    allows different regularisation strengths for each stage.

Network features expected (produced by FinanceNetworkBase.transform())
----------------------------------------------------------------------
    net_degree            node degree in the current graph snapshot
    net_degree_change     neighbourhood turnover:
                            1 - |N_{t-1} ∩ N_t| / max(1, |N_t|)
                          0 = stable, 1 = complete rewiring; NaN at t=0
    net_idw_log_RV1       IDW mean of neighbours' log_RV1
    net_idw_log_RV5       IDW mean of neighbours' log_RV5
    net_idw_log_RV22      IDW mean of neighbours' log_RV22

Missing net_* values (rows before the first graph snapshot) are imputed
with 0.0 --- a sensible prior when no graph information is available.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Shared feature lists
# ---------------------------------------------------------------------------

HAR_FEATURES: List[str] = [
    "log_RV1",
    "log_RV5",
    "log_RV10",
    "log_RV22",
    "log_neg_semi5",
    "log_pos_semi5",
]

NET_FEATURES: List[str] = [
    "net_degree",
    "net_degree_change",
    "net_idw_log_RV1",
    "net_idw_log_RV5",
    "net_idw_log_RV22",
]


def _fill_net(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN in network columns with 0.0.

    NaNs appear in rows before the first graph snapshot (no connectivity
    information yet).  Using 0 rather than mean imputation avoids introducing
    test-period statistics into training rows.
    """
    net_cols_present = [c for c in NET_FEATURES if c in X.columns]
    if net_cols_present:
        X = X.copy()
        X[net_cols_present] = X[net_cols_present].fillna(0.0)
    return X


# ---------------------------------------------------------------------------
# NetworkHARRegressor  (single-stage)
# ---------------------------------------------------------------------------

class NetworkHARRegressor(BaseEstimator, RegressorMixin):
    """
    Single-stage log-space Lasso regressing on HAR + network features.

    log_Y ~ log_RV1 + log_RV5 + log_RV10 + log_RV22
            + log_neg_semi5 + log_pos_semi5
            + net_degree + net_degree_change
            + net_idw_log_RV1 + net_idw_log_RV5 + net_idw_log_RV22

    A per-fold StandardScaler (embedded in a Pipeline) normalises all inputs
    before regularised regression, ensuring scale-invariance and no leakage.

    Parameters
    ----------
    lasso_alpha : float
        L1 penalty strength.  Lasso naturally zeroes out irrelevant network
        features when they add no predictive value.  Set to 0.0 for OLS.
    ridge_alpha : float
        L2 penalty.  Used when lasso_alpha == 0.0 and ridge_alpha > 0.0.
    """

    def __init__(
        self,
        lasso_alpha: float = 0.05,
        ridge_alpha: float = 0.0,
    ):
        self.lasso_alpha = lasso_alpha
        self.ridge_alpha = ridge_alpha
        self.features: List[str] = HAR_FEATURES + NET_FEATURES
        self._pipe: Optional[Pipeline] = None

    def _make_regressor(self):
        if self.lasso_alpha > 0:
            return Lasso(alpha=self.lasso_alpha, max_iter=20_000)
        if self.ridge_alpha > 0:
            return Ridge(alpha=self.ridge_alpha)
        return LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NetworkHARRegressor":
        X_fit = _fill_net(X)[self.features]
        self._pipe = Pipeline(
            [("scaler", StandardScaler()), ("reg", self._make_regressor())]
        )
        self._pipe.fit(X_fit, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_pred = _fill_net(X)[self.features]
        return np.exp(self._pipe.predict(X_pred))


# ---------------------------------------------------------------------------
# NetworkVARRegressor  (two-stage network error-correction)
# ---------------------------------------------------------------------------

class NetworkVARRegressor(BaseEstimator, RegressorMixin):
    """
    Two-stage Network Vector AutoRegression for weekly realised variance.

    Motivation
    ----------
    Standard HAR models only use each stock's own past volatility.  In a
    connected market, shocks propagate through the correlation network, so
    a stock's HAR residuals should be systematically related to its neighbours'
    volatility state.  This model captures that cross-sectional contagion term
    explicitly by fitting it in a separate stage.

    Stage 1 -- Own dynamics
        Fit a HAR-Extended OLS regression in log space using only own lags:
            log_Ŷ_1 = f(log_RV1, log_RV5, log_RV10, log_RV22,
                         log_neg_semi5, log_pos_semi5)
        Compute training residuals: e_t = log_Y_t - log_Ŷ_1,t

    Stage 2 -- Network error correction
        Fit a Ridge regression of e_t on the network features:
            ê_t = g(net_degree, net_degree_change,
                    net_idw_log_RV1, net_idw_log_RV5, net_idw_log_RV22)
        The correction is bounded to [-bound, +bound] in log space to prevent
        explosive predictions from noisy network features.

    Final prediction
        log_Ŷ = log_Ŷ_1 + clamp(ê, -bound, +bound)
        Ŷ     = exp(log_Ŷ)

    Parameters
    ----------
    stage2_alpha : float
        Ridge penalty for the network error-correction stage.  Higher values
        shrink the network correction towards zero.  Tune this if the network
        features are noisy or the graph is sparse.
    correction_bound : float
        Maximum absolute log-space correction from stage 2.  Acts as a safety
        valve against extreme network predictions.  A value of 0.5 means the
        network can shift the HAR prediction by at most e^0.5 ≈ 1.65× up or
        down in original scale.
    """

    def __init__(
        self,
        stage2_alpha: float = 0.1,
        correction_bound: float = 0.5,
    ):
        self.stage2_alpha = stage2_alpha
        self.correction_bound = correction_bound
        self.features: List[str] = HAR_FEATURES + NET_FEATURES

        self._stage1: Optional[Pipeline] = None
        self._stage2: Optional[Pipeline] = None

    # ── Stage 1: own HAR-Extended OLS ──────────────────────────────────

    def _fit_stage1(self, X: pd.DataFrame, log_y: pd.Series) -> pd.Series:
        """Fit stage-1 OLS; return in-sample log-space residuals."""
        self._stage1 = Pipeline(
            [("scaler", StandardScaler()), ("reg", LinearRegression())]
        )
        self._stage1.fit(X[HAR_FEATURES], log_y)
        log_pred1 = pd.Series(
            self._stage1.predict(X[HAR_FEATURES]), index=log_y.index
        )
        return log_y - log_pred1   # residuals

    # ── Stage 2: network residual correction ───────────────────────────

    def _fit_stage2(self, X: pd.DataFrame, residuals: pd.Series) -> None:
        """Fit stage-2 Ridge on network features using HAR residuals as target."""
        X_net = _fill_net(X)[NET_FEATURES]
        self._stage2 = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=self.stage2_alpha)),
            ]
        )
        self._stage2.fit(X_net, residuals)

    # ── sklearn API ────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NetworkVARRegressor":
        log_y = y.copy()   # y is already log_Y (see fit_predict_model)

        residuals = self._fit_stage1(X, log_y)
        self._fit_stage2(X, residuals)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_pred1 = self._stage1.predict(X[HAR_FEATURES])

        X_net = _fill_net(X)[NET_FEATURES]
        correction = self._stage2.predict(X_net)
        correction = np.clip(correction, -self.correction_bound, self.correction_bound)

        return np.exp(log_pred1 + correction)
