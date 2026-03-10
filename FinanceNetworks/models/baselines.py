from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Simple heuristic baselines (no fitting required)
# ---------------------------------------------------------------------------

def baseline_predict_naive_week(df_feat: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Repeat the past-week RV as the forecast."""
    return horizon * df_feat["RV5"]


def baseline_predict_roll22(df_feat: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Repeat the past-month RV as the forecast."""
    return horizon * df_feat["RV22"]


def baseline_predict_ewma(df_feat: pd.DataFrame, lam: float = 0.94, horizon: int = 5) -> pd.Series:
    """EWMA (RiskMetrics-style) on daily squared returns."""
    r2 = df_feat["RV1"].values
    sigma2 = np.empty_like(r2)
    sigma2[0] = np.nanmean(r2[:50]) if len(r2) >= 50 else np.nanmean(r2)
    for i in range(1, len(r2)):
        sigma2[i] = lam * sigma2[i - 1] + (1 - lam) * r2[i - 1]
    return pd.Series(horizon * sigma2, index=df_feat.index)


# ---------------------------------------------------------------------------
# HAR model (canonical 3-lag version)
# ---------------------------------------------------------------------------

def _build_regressor(
    ridge_alpha: float = 0.0,
    lasso_alpha: float = 0.0,
) -> "LinearRegression | Ridge | Lasso":
    """Return the appropriate sklearn regressor based on penalty parameters."""
    if lasso_alpha > 0:
        return Lasso(alpha=lasso_alpha, max_iter=10_000)
    if ridge_alpha > 0:
        return Ridge(alpha=ridge_alpha)
    return LinearRegression()


class HARLogRegressor(BaseEstimator, RegressorMixin):
    """
    HAR model in log space: log_Y ~ log_RV1 + log_RV5 + log_RV22.
    Predictions are returned in original variance scale via exp().

    Penalty selection (mutually exclusive; lasso takes priority):
      lasso_alpha > 0  → Lasso  (L1, sparse solution)
      ridge_alpha > 0  → Ridge  (L2, shrinkage)
      both == 0        → OLS

    Scaling:
        A StandardScaler is embedded inside a Pipeline and is fit only on the
        training slice of each cross-validation fold, preventing data leakage.
    """

    def __init__(self, ridge_alpha: float = 0.0, lasso_alpha: float = 0.0):
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.features = ["log_RV1", "log_RV5", "log_RV22"]
        self.model_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARLogRegressor":
        base = _build_regressor(self.ridge_alpha, self.lasso_alpha)
        self.model_ = Pipeline([("scaler", StandardScaler()), ("reg", base)])
        self.model_.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_yhat = self.model_.predict(X[self.features])
        return np.exp(log_yhat)


# ---------------------------------------------------------------------------
# HAR-Extended model (adds bi-weekly lag + asymmetric semivariance)
# ---------------------------------------------------------------------------

class HARExtendedLogRegressor(BaseEstimator, RegressorMixin):
    """
    Extended HAR model in log space:
        log_Y ~ log_RV1 + log_RV5 + log_RV10 + log_RV22
                + log_neg_semi5 + log_pos_semi5

    The asymmetric semi-variance terms capture the leverage effect: negative
    return days contribute disproportionately more to future volatility than
    positive return days of the same magnitude.

    Like HARLogRegressor, a per-fold StandardScaler is embedded in a Pipeline
    to avoid cross-validation data leakage.
    """

    def __init__(self, ridge_alpha: float = 0.0, lasso_alpha: float = 0.0):
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.features = [
            "log_RV1",
            "log_RV5",
            "log_RV10",
            "log_RV22",
            "log_neg_semi5",
            "log_pos_semi5",
        ]
        self.model_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARExtendedLogRegressor":
        base = _build_regressor(self.ridge_alpha, self.lasso_alpha)
        self.model_ = Pipeline([("scaler", StandardScaler()), ("reg", base)])
        self.model_.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_yhat = self.model_.predict(X[self.features])
        return np.exp(log_yhat)


# ---------------------------------------------------------------------------
# ARIMA on log_Y
# ---------------------------------------------------------------------------

class ARIMALogY:
    """
    ARIMA on log_Y; forecasts log_Y then exp() to get Y_fwd predictions.
    Uses statsmodels.  X is ignored during fit (univariate model).

    Note on the log_Y series: consecutive values share horizon-1 squared
    returns (overlapping windows), so the series is strongly autocorrelated
    by construction.  ARIMA can exploit this structure without any leakage
    because only training-set values of log_Y are passed to the fitter.

    Convergence robustness
    ----------------------
    Higher-order ARIMA models (p+q >= 2) can produce non-invertible MA roots
    or fail to converge on particular tickers.  We mitigate this by:
      1. Setting enforce_invertibility=True and enforce_stationarity=True so
         statsmodels keeps parameters inside the stable region.
      2. Trying a sequence of optimizers (lbfgs → powell → nm → bfgs) with
         increasing maxiter.  The first successful fit is kept.
      3. Falling back to AR(1) (always identifiable) if all optimizer
         attempts fail, so the CV loop is never interrupted.
    The ConvergenceWarning and Non-invertible MA warnings are suppressed
    because we handle them explicitly via the fallback logic.
    """

    # Optimizer sequence: fast quasi-Newton first, then gradient-free fallbacks
    _METHODS = [
        ("lbfgs",  500),
        ("powell", 1_000),
        ("nm",     2_000),
        ("bfgs",   2_000),
    ]
    _FALLBACK_ORDER = (1, 0, 0)

    def __init__(self, order: tuple = (1, 0, 1)):
        self.order = order
        self.res_ = None
        self.features = ["log_Y"]   # only the target column is needed for ARIMA

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import warnings
        from statsmodels.tsa.arima.model import ARIMA

        # Attach business-day frequency so statsmodels does not warn about gaps.
        y_fit = y.asfreq("B").ffill()

        def _try_fit(order, optimizer, maxiter):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(
                    y_fit,
                    order=order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(
                    method_kwargs={"method": optimizer, "maxiter": maxiter, "disp": False},
                )
            return res

        # Try requested order with each optimizer
        for method, maxiter in self._METHODS:
            try:
                self.res_ = _try_fit(self.order, method, maxiter)
                return self
            except Exception:
                continue

        # All optimizers failed -- fall back to AR(1)
        if self.order != self._FALLBACK_ORDER:
            import warnings
            warnings.warn(
                f"ARIMA{self.order} failed to converge on all optimizers; "
                f"falling back to AR(1).",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            self.res_ = _try_fit(self._FALLBACK_ORDER, "lbfgs", 500)
        except Exception as exc:
            raise RuntimeError(
                f"ARIMA fallback AR(1) also failed: {exc}"
            ) from exc

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_forecast = self.res_.forecast(steps=len(X))
        return np.exp(np.asarray(log_forecast))


# ---------------------------------------------------------------------------
# GARCH weekly RV forecast
# ---------------------------------------------------------------------------

class GARCHWeeklyRV:
    """
    Fits GARCH(p, q) on percentage log-returns, performs a rolling
    multi-step-ahead forecast over the test window, and sums the
    horizon-day conditional variances as the weekly RV prediction.

    Algorithm (per test row t):
      1. Observe r_t; compute eps2_t = r_t^2.
      2. Compute sigma2_{t+1} from the exact GARCH(p,q) recursion.
      3. Propagate E[sigma2_{t+k}] for k = 2..horizon analytically,
         substituting E[eps2] = E[sigma2] for future (mean-zero) steps.
      4. Sum E[sigma2_{t+1}]...E[sigma2_{t+horizon}] as the forecast.
      5. Append (eps2_t, sigma2_{t+1}) to the rolling history for the
         next step.

    Works for arbitrary p, q -- not just GARCH(1,1).
    Requires: pip install arch
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "zero",
        scale: float = 1.0,
        horizon: int = 5,
    ):
        self.p, self.q = p, q
        self.dist = dist
        self.mean = mean
        self.scale = scale
        self.horizon = horizon
        self.res_ = None
        # Requires these columns; log_RV* are present so the cross-val loop
        # can slice them without KeyError.
        self.features = ["log_RV1", "log_RV5", "log_RV22", "Returns"]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError(
                "arch package not available. Install via: pip install arch"
            ) from e

        x = X["Returns"].dropna() * self.scale
        am = arch_model(x, mean=self.mean, vol="GARCH", p=self.p, q=self.q, dist=self.dist)
        self.res_ = am.fit(disp="off")
        # Store scaled eps^2 and sigma^2 history for use in predict().
        self._eps2_hist   = list(self.res_.resid.values ** 2)
        self._sigma2_hist = list(self.res_.conditional_volatility.values ** 2)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        params = self.res_.params
        omega  = float(params["omega"])
        alphas = [float(params[f"alpha[{i}]"]) for i in range(1, self.p + 1)]
        betas  = [float(params[f"beta[{i}]"])  for i in range(1, self.q + 1)]

        eps2_hist   = list(self._eps2_hist)
        sigma2_hist = list(self._sigma2_hist)

        returns_test = X["Returns"].fillna(0).values * self.scale
        preds = np.empty(len(returns_test))

        for i, r in enumerate(returns_test):
            eps2_cur = r ** 2

            # 1-step-ahead: exact GARCH recursion
            eps2_with_cur = eps2_hist + [eps2_cur]
            sigma2_1 = (
                omega
                + sum(alphas[a] * eps2_with_cur[-(a + 1)] for a in range(self.p))
                + sum(betas[b]  * sigma2_hist[-(b + 1)]   for b in range(self.q))
            )

            # h-step-ahead: expectation propagation
            future_s2 = [sigma2_1]
            for k in range(1, self.horizon):
                s2_k = omega
                for a in range(self.p):
                    fut_idx = k - a
                    if fut_idx > 0:
                        s2_k += alphas[a] * future_s2[fut_idx - 1]
                    elif fut_idx == 0:
                        s2_k += alphas[a] * eps2_cur
                    else:
                        s2_k += alphas[a] * eps2_with_cur[fut_idx - 1]
                for b in range(self.q):
                    fut_idx = k - b
                    if fut_idx > 0:
                        s2_k += betas[b] * future_s2[fut_idx - 1]
                    elif fut_idx == 0:
                        s2_k += betas[b] * sigma2_1
                    else:
                        s2_k += betas[b] * sigma2_hist[fut_idx - 1]
                future_s2.append(s2_k)

            # Sum horizon conditional variances; undo scale^2
            preds[i] = sum(future_s2) / (self.scale ** 2)

            # Update rolling history
            eps2_hist.append(eps2_cur)
            sigma2_hist.append(sigma2_1)

        return preds


# ---------------------------------------------------------------------------
# EGARCH weekly RV forecast
# ---------------------------------------------------------------------------

class EGARCHWeeklyRV(BaseEstimator, RegressorMixin):
    """
    Faster EGARCH weekly-RV forecaster.

    Key design choice
    -----------------
     We fit EGARCH once on the training slice, then freeze the parameters and
     forecast the entire test block in one call by concatenating training and
     observed test returns.  This preserves the intended rolling-origin setup
     while avoiding an expensive model rebuild for every test row.

    Notes
    -----
    - Assumes X["Returns"] at row t is known at forecast origin t, and the target
      is future volatility from t+1 onward.
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
        n_simulations: int = 500,
    ):
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.mean = mean
        self.scale = scale
        self.horizon = horizon
        self.n_simulations = n_simulations
        self.res_ = None
        self.features = ["log_RV1", "log_RV5", "log_RV22", "Returns"]
        self._train_returns = None
        self._params = None
        self._arch_model_kwargs = {
            "mean": self.mean,
            "vol": "EGARCH",
            "p": self.p,
            "o": self.o,
            "q": self.q,
            "dist": self.dist,
        }

    def _make_model(self, series):
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError(
                "arch package not available. Install via: pip install arch"
            ) from e

        return arch_model(series, **self._arch_model_kwargs)

    def _prepare_training_returns(self, X: pd.DataFrame) -> np.ndarray:
        if "Returns" not in X.columns:
            raise ValueError("X must contain a 'Returns' column.")

        r = pd.to_numeric(X["Returns"], errors="coerce").dropna()
        if len(r) < max(50, self.horizon + 10):
            raise ValueError(
                f"Not enough non-missing training returns for EGARCH: {len(r)} rows."
            )

        return r.astype(float).values * self.scale

    def _prepare_predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        if "Returns" not in X.columns:
            raise ValueError("X must contain a 'Returns' column.")

        r_test = pd.to_numeric(X["Returns"], errors="coerce")
        if r_test.isna().any():
            bad = int(r_test.isna().sum())
            raise ValueError(
                f"X['Returns'] contains {bad} missing/non-numeric values. "
                "Please clean them before calling predict()."
            )

        return r_test.astype(float).values * self.scale

    def _fit_egarch(self, X: pd.DataFrame):
        x = self._prepare_training_returns(X)
        am = self._make_model(x)
        self.res_ = am.fit(disp="off")
        self._train_returns = x.copy()
        self._params = self.res_.params.copy()
        return self

    def _align_forecast_matrix(self, var_matrix: np.ndarray, n_rows: int) -> np.ndarray:
        if var_matrix.shape[0] == n_rows + 1:
            return var_matrix[1:]
        if var_matrix.shape[0] > n_rows:
            return var_matrix[-n_rows:]
        if var_matrix.shape[0] != n_rows:
            raise RuntimeError(
                f"Unexpected EGARCH forecast shape {var_matrix.shape}; expected {n_rows} rows."
            )
        return var_matrix

    def _forecast_in_sample(self) -> np.ndarray:
        if self.res_ is None:
            raise RuntimeError("Call fit() before requesting EGARCH forecasts.")

        fcst = self.res_.forecast(
            horizon=self.horizon,
            method="simulation",
            simulations=self.n_simulations,
            start=0,
            reindex=False,
        )
        var_matrix = np.asarray(fcst.variance.values, dtype=float)
        return var_matrix.sum(axis=1) / (self.scale ** 2)

    def _forecast_out_of_sample(self, X: pd.DataFrame) -> np.ndarray:
        if self.res_ is None or self._train_returns is None or self._params is None:
            raise RuntimeError("Call fit() before predict().")

        r_test = self._prepare_predict_returns(X)
        n_train = len(self._train_returns)
        n_pred = len(r_test)
        all_returns = np.concatenate([self._train_returns, r_test])

        fixed = self._make_model(all_returns).fix(self._params)
        fcst = fixed.forecast(
            horizon=self.horizon,
            method="simulation",
            simulations=self.n_simulations,
            start=n_train - 1,
            reindex=False,
        )

        var_matrix = np.asarray(fcst.variance.values, dtype=float)
        var_matrix = self._align_forecast_matrix(var_matrix, n_pred)
        return var_matrix.sum(axis=1) / (self.scale ** 2)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self._fit_egarch(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._forecast_out_of_sample(X)


# ---------------------------------------------------------------------------
# Regime-Switching HAR (threshold-based, 2 regimes)
# ---------------------------------------------------------------------------

class RegimeSwitchingHARLogRegressor(BaseEstimator, RegressorMixin):
    """
    Two-regime threshold HAR model in log space.

    Regime identification
    ---------------------
    Regimes are defined by the *regime_col* feature (default: log_RV22)
    relative to a threshold computed as the *regime_percentile* quantile of
    that column in the *training* fold:
        Regime 0 (low vol) : regime_col <= threshold
        Regime 1 (high vol): regime_col >  threshold

    Motivation: volatility dynamics differ markedly across calm and turbulent
    markets.  Fitting separate HAR equations per regime allows distinct
    persistence parameters, capturing:
      - Stronger mean-reversion in low-volatility regimes.
      - Slower decay and fatter coefficients in crisis regimes.

    A separate StandardScaler is embedded inside each regime's Pipeline so
    that training statistics are estimated only on the regime-specific subset
    of the training fold -- no look-ahead leakage.

    Parameters
    ----------
    ridge_alpha      : L2 penalty (forwarded to each regime's regressor).
    lasso_alpha      : L1 penalty (forwarded; lasso takes priority over ridge).
    regime_col       : Column used to define the regime indicator.
    regime_percentile: Training quantile used as the regime threshold.
                       0.5  -> median split.  Increase to make high-vol
                       regime rarer (e.g. 0.75 for top quartile).
    min_regime_obs   : Minimum number of observations a regime must have in
                       training before falling back to a pooled model.
    """

    def __init__(
        self,
        ridge_alpha: float = 0.0,
        lasso_alpha: float = 0.0,
        regime_col: str = "log_RV22",
        regime_percentile: float = 0.5,
        min_regime_obs: int = 30,
    ):
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.regime_col = regime_col
        self.regime_percentile = regime_percentile
        self.min_regime_obs = min_regime_obs
        self.features = ["log_RV1", "log_RV5", "log_RV22"]
        self.threshold_: float = 0.0
        self.models_: dict = {}
        self.fallback_model_: Optional[Pipeline] = None

    def _make_pipeline(self) -> Pipeline:
        base = _build_regressor(self.ridge_alpha, self.lasso_alpha)
        return Pipeline([("scaler", StandardScaler()), ("reg", base)])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeSwitchingHARLogRegressor":
        self.threshold_ = float(np.nanquantile(X[self.regime_col], self.regime_percentile))

        # Fit a pooled fallback model on all training data
        self.fallback_model_ = self._make_pipeline()
        self.fallback_model_.fit(X[self.features], y)

        # Fit per-regime models if each regime has enough observations
        for regime in (0, 1):
            mask = (
                X[self.regime_col] <= self.threshold_
                if regime == 0
                else X[self.regime_col] > self.threshold_
            )
            X_r, y_r = X.loc[mask, self.features], y.loc[mask]
            if len(X_r) >= self.min_regime_obs:
                m = self._make_pipeline()
                m.fit(X_r, y_r)
                self.models_[regime] = m

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.empty(len(X))

        for regime in (0, 1):
            mask = (
                (X[self.regime_col] <= self.threshold_)
                if regime == 0
                else (X[self.regime_col] > self.threshold_)
            )
            if not mask.any():
                continue
            model = self.models_.get(regime, self.fallback_model_)
            log_yhat = model.predict(X.loc[mask, self.features])
            preds[np.where(mask)[0]] = np.exp(log_yhat)

        return preds