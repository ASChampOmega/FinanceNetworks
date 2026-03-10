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

NetworkEGARCHRegressor
    Two-stage stacked EGARCH model.  Stage 1 produces an efficient EGARCH
    weekly-RV forecast; Stage 2 learns a bounded residual correction from
    network features only.

NetworkEGARCHXRegressor
    EGARCHX-style extension of NetworkEGARCHRegressor.  Uses the same EGARCH
    base forecast but exposes own-stock HAR features together with the network
    features in the correction stage.

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
    net_idw_Returns       IDW mean of neighbours' signed returns

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

from models.baselines import EGARCHWeeklyRV


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
    "Returns",          # signed return: captures leverage effect (negative
                        # shocks drive more future volatility than positive ones)
]

NET_FEATURES: List[str] = [
    "net_degree",
    "net_degree_change",
    "net_idw_log_RV1",
    "net_idw_log_RV5",
    "net_idw_log_RV22",
    "net_idw_Returns",  # IDW mean of neighbours' signed returns: cross-sectional
                        # leverage signal -- negative neighbour returns predict
                        # higher own future volatility via contagion
]

# Extended network features (clustering + market-wide connectivity)
CLUSTERING_FEATURES: List[str] = [
    "net_node_clustering",
    "net_global_clustering",
    "net_avg_abs_corr",
]

# All network features combined
NET_FEATURES_FULL: List[str] = NET_FEATURES + CLUSTERING_FEATURES


def _fill_net(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN in network columns with 0.0.

    NaNs appear in rows before the first graph snapshot (no connectivity
    information yet).  Using 0 rather than mean imputation avoids introducing
    test-period statistics into training rows.
    """
    all_net = NET_FEATURES_FULL
    net_cols_present = [c for c in all_net if c in X.columns]
    if net_cols_present:
        X = X.copy()
        X[net_cols_present] = X[net_cols_present].fillna(0.0)
    return X


def _select_net_features(use_clustering: bool) -> List[str]:
    """Return the appropriate network feature list."""
    return NET_FEATURES_FULL if use_clustering else NET_FEATURES


def _dedupe_preserve_order(columns: List[str]) -> List[str]:
    """Remove duplicate column names while preserving their first appearance."""
    return list(dict.fromkeys(columns))


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
        use_clustering: bool = False,
    ):
        self.lasso_alpha = lasso_alpha
        self.ridge_alpha = ridge_alpha
        self.use_clustering = use_clustering
        net_feats = _select_net_features(use_clustering)
        self.features: List[str] = HAR_FEATURES + net_feats
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
                         log_neg_semi5, log_pos_semi5, Returns)
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
        use_clustering: bool = False,
    ):
        self.stage2_alpha = stage2_alpha
        self.correction_bound = correction_bound
        self.use_clustering = use_clustering
        self._net_feats: List[str] = _select_net_features(use_clustering)
        self.features: List[str] = HAR_FEATURES + self._net_feats

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
        X_net = _fill_net(X)[self._net_feats]
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

        X_net = _fill_net(X)[self._net_feats]
        correction = self._stage2.predict(X_net)
        correction = np.clip(correction, -self.correction_bound, self.correction_bound)

        return np.exp(log_pred1 + correction)


# ---------------------------------------------------------------------------
# Stacked NetworkEGARCH models
# ---------------------------------------------------------------------------

class _BaseNetworkEGARCHRegressor(EGARCHWeeklyRV):
    """Shared two-stage EGARCH base + exogenous residual-correction model."""

    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "zero",
        scale: float = 1.0,
        horizon: int = 5,
        n_simulations: int = 500,
        stage2_alpha: float = 0.1,
        correction_bound: float = 0.5,
        use_clustering: bool = False,
    ):
        super().__init__(
            p=p,
            o=o,
            q=q,
            dist=dist,
            mean=mean,
            scale=scale,
            horizon=horizon,
            n_simulations=n_simulations,
        )
        self.stage2_alpha = stage2_alpha
        self.correction_bound = correction_bound
        self.use_clustering = use_clustering
        self._net_feats: List[str] = _select_net_features(use_clustering)
        self.features = _dedupe_preserve_order(["Returns"] + self._stage2_feature_names())
        self._stage2: Optional[Pipeline] = None

    def _stage2_feature_names(self) -> List[str]:
        raise NotImplementedError

    def _make_stage2_regressor(self):
        if self.stage2_alpha > 0:
            return Ridge(alpha=self.stage2_alpha)
        return LinearRegression()

    def _build_stage2_matrix(self, X: pd.DataFrame, log_base: np.ndarray) -> pd.DataFrame:
        X_stage = _fill_net(X)
        extra_cols = self._stage2_feature_names()
        if extra_cols:
            stage_df = X_stage[extra_cols].copy()
        else:
            stage_df = pd.DataFrame(index=X.index)
        stage_df.insert(0, "log_egarch_base", np.asarray(log_base, dtype=float))
        return stage_df

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BaseNetworkEGARCHRegressor":
        super().fit(X, y)

        base_train = self._forecast_in_sample()
        log_base_train = np.log(np.clip(base_train, 1e-12, None))
        residuals = y.values - log_base_train

        self._stage2 = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", self._make_stage2_regressor()),
            ]
        )
        self._stage2.fit(self._build_stage2_matrix(X, log_base_train), residuals)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._stage2 is None:
            raise RuntimeError("Call fit() before predict().")

        base_test = self._forecast_out_of_sample(X)
        log_base_test = np.log(np.clip(base_test, 1e-12, None))
        correction = self._stage2.predict(self._build_stage2_matrix(X, log_base_test))
        correction = np.clip(correction, -self.correction_bound, self.correction_bound)
        return np.exp(log_base_test + correction)


class NetworkEGARCHRegressor(_BaseNetworkEGARCHRegressor):
    """
    Efficient network-augmented EGARCH forecaster.

    Stage 1
        Fit EGARCH once on the training returns and produce the usual weekly
        RV forecasts for both train and test windows.

    Stage 2
        Learn a bounded log-space correction using only network features plus
        the base EGARCH forecast.  This keeps the expensive volatility fitting
        unchanged while allowing the correlation-network state to shift the
        final prediction.
    """

    def _stage2_feature_names(self) -> List[str]:
        return self._net_feats


class NetworkEGARCHXRegressor(_BaseNetworkEGARCHRegressor):
    """
    EGARCHX-style stacked forecaster using own HAR and network exogenous terms.

    This is an efficient approximation to a full joint EGARCHX estimation:
    the EGARCH dynamics are fit once in stage 1, and a second-stage linear
    correction uses the base EGARCH forecast together with exogenous features.
    """

    def _stage2_feature_names(self) -> List[str]:
        return _dedupe_preserve_order(HAR_FEATURES + self._net_feats)
