"""
models/baselines_classification.py
===================================
Logistic-regression classification wrappers for volatility-spike prediction.

Each classifier mirrors its regression counterpart in baselines.py but
replaces the linear regressor with LogisticRegression and exposes both
``predict`` (hard labels) and ``predict_proba`` (class probabilities needed
for ROC-AUC).

Classes
-------
HARLogitClassifier          : Plain logistic regression on [log_RV1, log_RV5, log_RV22].
HARExtendedLogitClassifier  : Adds log_RV10, log_neg_semi5, log_pos_semi5.
DCCGARCHSpikeClassifier     : DCC-GARCH volatility score + logistic calibration.

Both classes
  - embed a per-fold StandardScaler in a Pipeline (no look-ahead scaling),
  - accept a regularisation parameter C (inverse of ridge penalty; lower C
    means stronger L2 regularisation),
  - expose a ``features`` attribute so the CV loop can slice the right columns
    from the feature DataFrame (same convention as baselines.py).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import DCCGARCHWeeklyRV


# ---------------------------------------------------------------------------
# HAR logistic classifier
# ---------------------------------------------------------------------------

class HARLogitClassifier(BaseEstimator, ClassifierMixin):
    """
    Logistic regression on HAR log-RV features for spike classification.

    Model
    -----
    P(spike = 1 | log_RV1, log_RV5, log_RV22) via L2-penalised logistic
    regression with strength 1/C.

    Scaling
    -------
    A StandardScaler is fit *inside* a Pipeline on the training fold only,
    preventing data leakage across cross-validation splits.

    Parameters
    ----------
    C        : Inverse regularisation strength (sklearn convention).
               Smaller C → stronger regularisation.
    max_iter : Maximum solver iterations (default 1 000; LBFGs converges
               quickly on standardised features).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1_000, use_market: bool = True):
        self.C = C
        self.max_iter = max_iter
        self.use_market = use_market
        self.features = ["log_RV1", "log_RV5", "log_RV22"]
        if use_market:
            self.features += ["Market_Returns", "log_Market_RV5", "log_Market_RV22"]
        self.model_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARLogitClassifier":
        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",

        )
        self.model_ = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.model_.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[self.features])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.features])


# ---------------------------------------------------------------------------
# HAR-Extended logistic classifier
# ---------------------------------------------------------------------------

class HARExtendedLogitClassifier(BaseEstimator, ClassifierMixin):
    """
    Logistic regression on extended HAR log-RV features for spike classification.

    Model
    -----
    P(spike = 1 | log_RV1, log_RV5, log_RV10, log_RV22,
                  log_neg_semi5, log_pos_semi5)

    The asymmetric semivariance terms (log_neg_semi5, log_pos_semi5) capture
    the leverage effect: negative-return days inflate future volatility more
    than positive days of equal magnitude, so they are valuable predictors of
    an upcoming spike.

    Parameters
    ----------
    C        : Inverse regularisation strength.
    max_iter : Maximum solver iterations.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1_000, use_market: bool = True):
        self.C = C
        self.max_iter = max_iter
        self.use_market = use_market
        self.features = [
            "log_RV1",
            "log_RV5",
            "log_RV10",
            "log_RV22",
            "log_neg_semi5",
            "log_pos_semi5",
        ]
        if use_market:
            self.features += ["Market_Returns", "log_Market_RV5", "log_Market_RV22"]
        self.model_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARExtendedLogitClassifier":
        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",

        )
        self.model_ = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.model_.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[self.features])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.features])


# ---------------------------------------------------------------------------
# Regime-Switching HAR logistic classifier (threshold-based, 2 regimes)
# ---------------------------------------------------------------------------

class RegimeSwitchingHARLogitClassifier(BaseEstimator, ClassifierMixin):
    """
    Two-regime threshold HAR logistic classifier for volatility-spike prediction.

    Regime identification
    ---------------------
    Identical to RegimeSwitchingHARLogRegressor: the training *regime_col*
    quantile at *regime_percentile* defines the threshold.
        Regime 0 (low vol) : regime_col <= threshold
        Regime 1 (high vol): regime_col >  threshold

    Motivation
    ----------
    The probability of a vol spike is much higher in the high-volatility
    regime.  Splitting the data and fitting regime-specific logistic
    regressions lets the classifier learn:
      * Regime 0: milder coefficients, lower baseline spike probability.
      * Regime 1: steeper coefficients, higher baseline spike probability.
    A single pooled logistic regression smears both regimes together,
    underestimating the spike probability in crisis periods.

    A per-regime StandardScaler is fit only on the regime-specific subset of
    training data, preventing cross-regime leakage.

    Parameters
    ----------
    C                : Inverse regularisation strength (per-regime logistic).
    max_iter         : Maximum logistic solver iterations.
    regime_col       : Feature column that drives the regime indicator.
    regime_percentile: Quantile of regime_col used as the split threshold.
    min_regime_obs   : Minimum training obs per regime.  Regimes with fewer
                       observations fall back to the pooled logistic model.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1_000,
        regime_col: str = "log_RV22",
        regime_percentile: float = 0.5,
        min_regime_obs: int = 30,
        use_market: bool = True,
    ):
        self.C = C
        self.max_iter = max_iter
        self.regime_col = regime_col
        self.regime_percentile = regime_percentile
        self.min_regime_obs = min_regime_obs
        self.use_market = use_market
        self.features = ["log_RV1", "log_RV5", "log_RV22"]
        if use_market:
            self.features += ["Market_Returns", "log_Market_RV5", "log_Market_RV22"]
        self.threshold_: float = 0.0
        self.models_: dict = {}
        self.fallback_model_: Optional[Pipeline] = None

    def _make_pipeline(self) -> Pipeline:
        clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs",

        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeSwitchingHARLogitClassifier":
        self.threshold_ = float(np.nanquantile(X[self.regime_col], self.regime_percentile))

        # Pooled fallback trained on all data
        self.fallback_model_ = self._make_pipeline()
        self.fallback_model_.fit(X[self.features], y)

        for regime in (0, 1):
            mask = (
                X[self.regime_col] <= self.threshold_
                if regime == 0
                else X[self.regime_col] > self.threshold_
            )
            X_r, y_r = X.loc[mask, self.features], y.loc[mask]
            if len(X_r) >= self.min_regime_obs and len(y_r.unique()) > 1:
                m = self._make_pipeline()
                m.fit(X_r, y_r)
                self.models_[regime] = m

        return self

    def _regime_mask(self, X: pd.DataFrame, regime: int):
        return (
            X[self.regime_col] <= self.threshold_
            if regime == 0
            else X[self.regime_col] > self.threshold_
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.empty(len(X), dtype=int)
        for regime in (0, 1):
            mask = self._regime_mask(X, regime)
            if not mask.any():
                continue
            model = self.models_.get(regime, self.fallback_model_)
            preds[np.where(mask)[0]] = model.predict(X.loc[mask, self.features])
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Determine number of classes from fallback model
        n_classes = len(self.fallback_model_.classes_)
        probas = np.zeros((len(X), n_classes))
        for regime in (0, 1):
            mask = self._regime_mask(X, regime)
            if not mask.any():
                continue
            model = self.models_.get(regime, self.fallback_model_)
            probas[np.where(mask)[0]] = model.predict_proba(X.loc[mask, self.features])
        return probas


# ---------------------------------------------------------------------------
# DCC-GARCH spike classifier
# ---------------------------------------------------------------------------

class DCCGARCHSpikeClassifier(BaseEstimator, ClassifierMixin):
    """
    Spike classifier built on top of DCCGARCHWeeklyRV forecasts.

    Workflow
    --------
    1. Fit DCC-GARCH on training returns.
    2. Build in-sample one-step systemic-variance scores.
    3. Fit a 1D logistic calibration from score -> spike probability.

    This keeps the API aligned with the existing classification CV pipeline
    while using a volatility-model signal instead of HAR regression features.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "zero",
        scale: float = 1.0,
        aux_returns_col: Optional[str] = "Market_Returns",
        rho_weight: float = 0.5,
        C: float = 1.0,
        max_iter: int = 1_000,
    ):
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self.scale = scale
        self.aux_returns_col = aux_returns_col
        self.rho_weight = rho_weight
        self.C = C
        self.max_iter = max_iter

        self.features = ["log_RV1", "log_RV5", "log_RV22", "Returns"]
        if aux_returns_col is not None:
            self.features.append(aux_returns_col)

        self.dcc_: Optional[DCCGARCHWeeklyRV] = None
        self.calibrator_: Optional[Pipeline] = None

    def _score_from_weekly_var(self, weekly_var: np.ndarray) -> np.ndarray:
        x = np.log(np.clip(weekly_var, 1e-12, None))
        return x.reshape(-1, 1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DCCGARCHSpikeClassifier":
        self.dcc_ = DCCGARCHWeeklyRV(
            p=self.p,
            q=self.q,
            dist=self.dist,
            mean=self.mean,
            scale=self.scale,
            horizon=1,
            aux_returns_col=self.aux_returns_col,
            rho_weight=self.rho_weight,
        )
        self.dcc_.fit(X, y)

        # In-sample rolling scores for calibration.
        in_sample_weekly = self.dcc_.predict(X)
        x_score = self._score_from_weekly_var(in_sample_weekly)

        clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs",

        )
        self.calibrator_ = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.calibrator_.fit(x_score, y.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        weekly_var = self.dcc_.predict(X)
        x_score = self._score_from_weekly_var(weekly_var)
        return self.calibrator_.predict_proba(x_score)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
