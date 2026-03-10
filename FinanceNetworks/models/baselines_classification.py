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

    def __init__(self, C: float = 1.0, max_iter: int = 1_000):
        self.C = C
        self.max_iter = max_iter
        # Feature set matches the canonical HAR regression model
        self.features = ["log_RV1", "log_RV5", "log_RV22"]
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

    def __init__(self, C: float = 1.0, max_iter: int = 1_000):
        self.C = C
        self.max_iter = max_iter
        # Feature set mirrors HARExtendedLogRegressor from baselines.py
        self.features = [
            "log_RV1",
            "log_RV5",
            "log_RV10",
            "log_RV22",
            "log_neg_semi5",
            "log_pos_semi5",
        ]
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
# EGARCH classifier
# ---------------------------------------------------------------------------

class EGARCHClassifier(BaseEstimator, ClassifierMixin):
    """
    Volatility-spike classifier built on top of EGARCH variance forecasts.

    Approach
    --------
    1. Fit EGARCH(p, o, q) on training returns (same as EGARCHWeeklyRV).
    2. Compute the simulation-based horizon-sum variance forecast for every
       *training* observation and take its log as a single derived feature.
    3. Calibrate a logistic regression that maps log(EGARCH_forecast) to the
       binary spike label.  A StandardScaler is embedded in the Pipeline.
    4. At test time, produce the EGARCH forecast for each test observation
       (using fixed training params + updated volatility filter), then pass
       log(forecast) through the calibrated logistic regression.

    Benefits
    --------
    * Inherits the leverage-effect asymmetry of EGARCH.
    * Logistic calibration maps the continuously-valued forecast to a proper
      probability without hard-coding a threshold -- the threshold is learned
      from the training label distribution.
    * predict_proba() returns well-calibrated probabilities suitable for
      ROC-AUC evaluation.

    Parameters
    ----------
    p, o, q       : EGARCH orders (see EGARCHWeeklyRV for details).
    dist          : Innovation distribution for EGARCH.
    mean          : Mean model (\"zero\" recommended for log-returns).
    scale         : Return scaling factor applied before EGARCH fitting.
    horizon       : Forecast horizon in days (should match spike definition).
    n_simulations : Monte-Carlo draws per forecast origin.
    C             : Inverse regularisation strength for the logistic layer.
    max_iter      : Maximum solver iterations for logistic regression.
    Requires      : pip install arch
    """

    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "zero",
        scale: float = 1.0,
        horizon: int = 5,
        n_simulations: int = 200,
        C: float = 1.0,
        max_iter: int = 1_000,
    ):
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.mean = mean
        self.scale = scale
        self.horizon = horizon
        self.n_simulations = n_simulations
        self.C = C
        self.max_iter = max_iter
        self.features = ["log_RV1", "log_RV5", "log_RV22", "Returns"]
        self._egarch_res = None
        self.logit_: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    # Internal: shared EGARCH forecast logic (mirrors EGARCHWeeklyRV)
    # ------------------------------------------------------------------

    def _fit_egarch(self, X: pd.DataFrame):
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError("pip install arch") from e

        x = X["Returns"].dropna() * self.scale
        am = arch_model(
            x, mean=self.mean, vol="EGARCH",
            p=self.p, o=self.o, q=self.q, dist=self.dist,
        )
        self._egarch_res = am.fit(disp="off")
        self._train_returns = x.values.copy()

    def _egarch_forecast(self, X: pd.DataFrame) -> np.ndarray:
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError("pip install arch") from e

        returns = X["Returns"].fillna(0).values * self.scale
        n_train = len(self._train_returns)
        n_pred  = len(returns)

        all_returns = pd.Series(
            np.concatenate([self._train_returns, returns]), dtype=float
        )
        am_full = arch_model(
            all_returns, mean=self.mean, vol="EGARCH",
            p=self.p, o=self.o, q=self.q, dist=self.dist,
        )
        fixed = am_full.fix(self._egarch_res.params)
        fcsts = fixed.forecast(
            horizon=self.horizon,
            start=n_train - 1,
            method="simulation",
            simulations=self.n_simulations,
            reindex=False,
        )
        var_matrix = fcsts.variance.values
        if var_matrix.shape[0] == n_pred + 1:
            var_matrix = var_matrix[1:]
        elif var_matrix.shape[0] > n_pred:
            var_matrix = var_matrix[-n_pred:]

        return var_matrix.sum(axis=1) / (self.scale ** 2)

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EGARCHClassifier":
        self._fit_egarch(X)

        # Derive in-sample EGARCH horizon-sum forecasts using the fitted result
        # directly.  Calling _egarch_forecast(X_train) would concatenate training
        # returns with themselves (doubling the series), creating a mismatch
        # between the EGARCH states used for logistic calibration (from the
        # "second pass" through training data) and those used at predict() time
        # (first pass through test data initialized from end-of-training state).
        # Using self._egarch_res.forecast(start=0) gives the true in-sample
        # conditional-variance forecasts without any doubling artefact.
        in_sample_fcsts = self._egarch_res.forecast(
            horizon=self.horizon,
            method="simulation",
            simulations=self.n_simulations,
            start=0,
            reindex=False,
        )
        var_matrix = np.asarray(in_sample_fcsts.variance.values, dtype=float)
        train_forecast = var_matrix.sum(axis=1) / (self.scale ** 2)

        log_fcst = np.log(np.clip(train_forecast, 1e-12, None)).reshape(-1, 1)

        clf = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        self.logit_ = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.logit_.fit(log_fcst, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        test_forecast = self._egarch_forecast(X)
        log_fcst = np.log(np.clip(test_forecast, 1e-12, None)).reshape(-1, 1)
        return self.logit_.predict(log_fcst)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        test_forecast = self._egarch_forecast(X)
        log_fcst = np.log(np.clip(test_forecast, 1e-12, None)).reshape(-1, 1)
        return self.logit_.predict_proba(log_fcst)


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
    ):
        self.C = C
        self.max_iter = max_iter
        self.regime_col = regime_col
        self.regime_percentile = regime_percentile
        self.min_regime_obs = min_regime_obs
        self.features = ["log_RV1", "log_RV5", "log_RV22"]
        self.threshold_: float = 0.0
        self.models_: dict = {}
        self.fallback_model_: Optional[Pipeline] = None

    def _make_pipeline(self) -> Pipeline:
        clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs"
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
