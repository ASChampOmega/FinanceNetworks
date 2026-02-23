from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge

# Predict the last week
def baseline_predict_naive_week(df_feat: pd.DataFrame, horizon: int = 5) -> pd.Series:
    return horizon * df_feat["RV5"]

# Predict the last month
def baseline_predict_roll22(df_feat: pd.DataFrame, horizon: int = 5) -> pd.Series:
    return horizon * df_feat["RV22"]

# Use EWMA on squared returns (RV1)
def baseline_predict_ewma(df_feat: pd.DataFrame, lam: float = 0.94, horizon: int = 5) -> pd.Series:
    r2 = df_feat["RV1"].values
    sigma2 = np.empty_like(r2)
    sigma2[0] = np.nanmean(r2[:50]) if len(r2) >= 50 else np.nanmean(r2)
    for i in range(1, len(r2)):
        sigma2[i] = lam * sigma2[i-1] + (1 - lam) * r2[i-1]
    return pd.Series(horizon * sigma2, index=df_feat.index)

# def fit_predict_har_log(
#     df_feat: pd.DataFrame,
#     train_idx: pd.Index,
#     test_idx: pd.Index,
# ) -> pd.Series:
#     """
#     HAR: log_Y ~ const + log_RV1 + log_RV5 + log_RV22
#     Fit on train, predict on test.
#     """
#     import statsmodels.api as sm

#     X_cols = ["log_RV1", "log_RV5", "log_RV22"]

#     X_train = sm.add_constant(df_feat.loc[train_idx, X_cols])
#     y_train = df_feat.loc[train_idx, "log_Y"]

#     model = sm.OLS(y_train, X_train, missing="drop").fit()

#     X_test = sm.add_constant(df_feat.loc[test_idx, X_cols], has_constant="add")
#     log_yhat = model.predict(X_test)

#     # Return predictions in original scale (variance)
#     yhat = np.exp(log_yhat)
#     return pd.Series(yhat, index=test_idx), model

class HARLogRegressor(BaseEstimator, RegressorMixin):
    """
    HAR model in log space: log_Y ~ log_RV1 + log_RV5 + log_RV22
    Predicts Y on original scale via exp().
    """
    def __init__(self, ridge_alpha: float = 0.0):
        self.ridge_alpha = ridge_alpha
        self.model_ = None
        self.features = ["log_RV1", "log_RV5", "log_RV22"]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # X includes the HAR columns already
        if self.ridge_alpha > 0:
            self.model_ = Ridge(alpha=self.ridge_alpha)
        else:
            self.model_ = LinearRegression()
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_yhat = self.model_.predict(X)
        return np.exp(log_yhat)

class ARIMALogY:
    """
    ARIMA on log_Y; forecasts log_Y then exp() to get Y.
    Uses statsmodels.
    For finding the order, we should run a GridSearch.
    features = ["log_Y"] — only y is used for fitting; X is ignored.
    """
    def __init__(self, order=(1,0,1)):
        self.order = order
        self.res_ = None
        self.features = ["log_Y"]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA
        # Attach business-day frequency so statsmodels doesn't warn about a missing freq.
        y_fit = y.asfreq("B").ffill()
        self.res_ = ARIMA(y_fit, order=self.order).fit()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_forecast = self.res_.forecast(steps=len(X))
        return np.exp(np.asarray(log_forecast))

class GARCHWeeklyRV:
    """
    Fits GARCH/ARCH on returns, forecasts next-horizon conditional variances,
    sums them to approximate weekly RV, and broadcasts the single forecast
    across all test rows as a constant baseline.
    Requires: pip install arch
    features = ["Returns"] — only returns are used for fitting.
    """
    def __init__(self, p=1, q=1, dist="normal", mean="zero", scale=1.0, horizon=5):
        self.p, self.q = p, q
        self.dist = dist
        self.mean = mean
        self.scale = scale
        self.horizon = horizon
        self.res_ = None
        self.features = ["log_RV1", "log_RV5", "log_RV22", "Returns"]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            from arch import arch_model
        except Exception as e:
            raise ImportError("arch package not available. Install via: pip install arch") from e

        x = X["Returns"].dropna() * self.scale
        am = arch_model(x, mean=self.mean, vol="GARCH", p=self.p, q=self.q, dist=self.dist)
        self.res_ = am.fit(disp="off")
        # Store scaled eps^2 and sigma^2 history for use in predict()
        self._eps2_hist = list(self.res_.resid.values ** 2)
        self._sigma2_hist = list(self.res_.conditional_volatility.values ** 2)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Rolling multi-step GARCH(p,q) forecast over the test window.

        At each test step t we:
          1. Observe the actual return r_t and compute eps2_t = r_t^2.
          2. Compute sigma2_{t+1} exactly from the full GARCH(p,q) recursion.
          3. Propagate E[sigma2_{t+k}] forward for k=2..horizon analytically,
             using E[eps2_{t+k}] = E[sigma2_{t+k}] for future (mean-zero) steps
             and observed values for current/past steps.
          4. Sum E[sigma2_{t+1..t+horizon}] as the weekly RV forecast.
        Works for any p and q, not just GARCH(1,1).
        """
        params = self.res_.params
        omega  = float(params["omega"])
        alphas = [float(params[f"alpha[{i}]"]) for i in range(1, self.p + 1)]
        betas  = [float(params[f"beta[{i}]"]) for i in range(1, self.q + 1)]

        # Mutable history lists (we append as we roll through the test window)
        eps2_hist   = list(self._eps2_hist)    # scaled units
        sigma2_hist = list(self._sigma2_hist)  # scaled units

        returns_test = X["Returns"].fillna(0).values * self.scale
        preds = np.empty(len(returns_test))

        for i, r in enumerate(returns_test):
            eps2_cur = r ** 2

            # ── 1-step-ahead: exact recursion ──────────────────────────────
            # alpha lags: eps2_cur is lag-0, eps2_hist[-1] is lag-1, etc.
            eps2_with_cur = eps2_hist + [eps2_cur]   # index -1 = eps2_cur
            sigma2_1 = (
                omega
                + sum(alphas[a] * eps2_with_cur[-(a + 1)] for a in range(self.p))
                + sum(betas[b]  * sigma2_hist[-(b + 1)]   for b in range(self.q))
            )

            # ── h-step-ahead: expectation propagation ──────────────────────
            # future_s2[k] = E[sigma2_{t+k+1}], k=0 means 1-step-ahead
            future_s2 = [sigma2_1]

            for k in range(1, self.horizon):
                s2_k = omega
                # Alpha terms: lags 1..p relative to step k+1
                for a in range(self.p):
                    fut_idx = k - a  # >0 → future, ==0 → current step, <0 → observed past
                    if fut_idx > 0:
                        # E[eps2] = E[sigma2] for future mean-zero steps
                        s2_k += alphas[a] * future_s2[fut_idx - 1]
                    elif fut_idx == 0:
                        s2_k += alphas[a] * eps2_cur
                    else:
                        s2_k += alphas[a] * eps2_with_cur[fut_idx - 1]
                # Beta terms: lags 1..q relative to step k+1
                for b in range(self.q):
                    fut_idx = k - b
                    if fut_idx > 0:
                        s2_k += betas[b] * future_s2[fut_idx - 1]
                    elif fut_idx == 0:
                        s2_k += betas[b] * sigma2_1
                    else:
                        s2_k += betas[b] * sigma2_hist[fut_idx - 1]
                future_s2.append(s2_k)

            # Sum horizon-day variance, convert from scaled^2 back to original units
            preds[i] = sum(future_s2) / (self.scale ** 2)

            # Update history for next step
            eps2_hist.append(eps2_cur)
            sigma2_hist.append(sigma2_1)

        return preds
