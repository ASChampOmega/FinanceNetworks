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
