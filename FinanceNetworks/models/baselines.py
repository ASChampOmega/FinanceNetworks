from __future__ import annotations
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

    def __init__(self, ridge_alpha: float = 0.0, lasso_alpha: float = 0.0, use_market: bool = True):
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.use_market = use_market
        self.features = ["log_RV1", "log_RV5", "log_RV22"]
        if use_market:
            self.features += ["Market_Returns", "log_Market_RV5", "log_Market_RV22"]
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

    def __init__(self, ridge_alpha: float = 0.0, lasso_alpha: float = 0.0, use_market: bool = True):
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
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
        returns_multiplier: float = 1.0,
    ):
        self.p, self.q = p, q
        self.dist = dist
        self.mean = mean
        self.scale = scale
        self.horizon = horizon
        self.returns_multiplier = returns_multiplier
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
        # Store training returns and fitted params for efficient vectorised predict().
        self._train_returns = x.values.copy()
        self._params = self.res_.params.copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Concatenate training + test returns, fix the fitted parameters, and
        # call arch's optimised analytic multi-step-ahead forecast in one shot.
        # This replaces the old hand-written Python loop (O(T) interpreter
        # iterations) with arch's internal C/Cython GARCH recursion, which is
        # dramatically faster and holds the GIL far less.
        from arch import arch_model
        r_test = X["Returns"].fillna(0).values * self.scale
        all_returns = np.concatenate([self._train_returns, r_test])
        n_train = len(self._train_returns)
        n_pred = len(r_test)

        fixed = arch_model(
            all_returns, mean=self.mean, vol="GARCH",
            p=self.p, q=self.q, dist=self.dist,
        ).fix(self._params)
        fcst = fixed.forecast(
            horizon=self.horizon,
            method="analytic",
            start=n_train - 1,
            reindex=False,
        )

        var_matrix = np.asarray(fcst.variance.values, dtype=float)
        # arch may return n_pred or n_pred+1 rows depending on version; align.
        if var_matrix.shape[0] == n_pred + 1:
            var_matrix = var_matrix[1:]
        elif var_matrix.shape[0] > n_pred:
            var_matrix = var_matrix[-n_pred:]

        # Sum horizon-step variances and undo scaling.
        return var_matrix.sum(axis=1) / (self.scale ** 2) / (self.returns_multiplier ** 2)


# ---------------------------------------------------------------------------
# DCC-GARCH weekly RV forecast
# ---------------------------------------------------------------------------

class DCCGARCHWeeklyRV:
    """
    Bivariate DCC-GARCH baseline for weekly RV forecasting.

    Forecast output
    ---------------
    Returns a horizon-day sum of spillover-adjusted variance:

        RV_hat = sum_k sigma1_k^2 * exp(rho_weight * rho_k^2
                   * [log(sigma2_k^2/sigma2_unc^2)
                      - log(sigma1_k^2/sigma1_unc^2)])

    The bracket is the *residual* market volatility surprise: the market's
    proportional deviation from its long-run level minus the asset's own
    deviation.  When both spike together the residual is near zero and the
    forecast equals plain GARCH.  Only when market vol moves *more* (or
    less) than what the asset GARCH has already absorbed does the DCC
    channel add a correction.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "zero",
        scale: float = 1.0,
        horizon: int = 5,
        aux_returns_col: Optional[str] = "Market_Returns",
        rho_weight: float = 0.10,
        dcc_start: tuple[float, float] = (0.03, 0.95),
        returns_multiplier: float = 1.0,
    ):
        self.p = int(p)
        self.q = int(q)
        self.dist = dist
        self.mean = mean
        self.scale = float(scale)
        self.horizon = int(horizon)
        self.aux_returns_col = aux_returns_col
        self.rho_weight = float(rho_weight)
        self.dcc_start = tuple(map(float, dcc_start))
        self.returns_multiplier = float(returns_multiplier)

        self.features = ["log_RV1", "log_RV5", "log_RV22", "Returns"]
        if aux_returns_col is not None:
            self.features.append(aux_returns_col)

        self.res1_ = None
        self.res2_ = None
        self._params1 = None
        self._params2 = None
        self._train_r1 = None
        self._train_r2 = None
        self._qbar = None
        self._q_last = None
        self._dcc_ab = None
        self._rho_unc = None
        self._var1_unc = None
        self._var2_unc = None
        self._last_train_return = None

    @staticmethod
    def _ensure_symmetric(a: np.ndarray) -> np.ndarray:
        return 0.5 * (a + a.T)

    @staticmethod
    def _regularize_pd(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        a = 0.5 * (a + a.T)
        return a + eps * np.eye(a.shape[0])

    def _check_is_fitted(self) -> None:
        if self._params1 is None or self._params2 is None or self._q_last is None:
            raise RuntimeError("Call fit() before predict().")

    def _aux_series(
        self,
        X: pd.DataFrame,
        *,
        previous_return: Optional[float] = None,
    ) -> pd.Series:
        """
        Auxiliary return series.

        If aux_returns_col exists, use it directly.
        Otherwise use lagged Returns, but preserve continuity across
        train/test by injecting the last training return into the first
        test observation.
        """
        if self.aux_returns_col is not None and self.aux_returns_col in X.columns:
            return X[self.aux_returns_col].astype(float)

        r = X["Returns"].astype(float).shift(1)
        if previous_return is not None and len(r) > 0:
            r = r.copy()
            r.iloc[0] = previous_return
        return r

    def _clean_pair(
        self,
        r1: pd.Series,
        r2: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        d = pd.DataFrame({"r1": r1, "r2": r2}).dropna()
        return (
            d["r1"].to_numpy(dtype=float) * self.scale,
            d["r2"].to_numpy(dtype=float) * self.scale,
        )

    def _fit_dcc(
        self,
        z: np.ndarray,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        from scipy.optimize import minimize

        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != 2:
            raise ValueError("z must be an (n, 2) array of standardized residuals.")
        if len(z) < 2:
            raise ValueError("Need at least 2 observations to fit DCC.")

        qbar = np.cov(z.T)
        qbar = self._regularize_pd(np.asarray(qbar, dtype=float))

        def _negloglike(ab: np.ndarray) -> float:
            a, b = float(ab[0]), float(ab[1])

            if a < 0.0 or b < 0.0 or (a + b) >= 0.999:
                return 1e12

            q_t = qbar.copy()
            nll = 0.0

            for t in range(1, len(z)):
                z_prev = z[t - 1][:, None]
                q_t = (1.0 - a - b) * qbar + a * (z_prev @ z_prev.T) + b * q_t
                q_t = self._regularize_pd(q_t)

                d = np.sqrt(np.clip(np.diag(q_t), 1e-12, None))
                r_t = q_t / np.outer(d, d)
                r_t = self._regularize_pd(r_t)

                sign, logdet_r = np.linalg.slogdet(r_t)
                if sign <= 0.0 or not np.isfinite(logdet_r):
                    return 1e12

                z_t = z[t]
                try:
                    inv_r_z = np.linalg.solve(r_t, z_t)
                except np.linalg.LinAlgError:
                    return 1e12

                quad = float(z_t @ inv_r_z)
                norm2 = float(z_t @ z_t)
                nll += 0.5 * (logdet_r + quad - norm2)

            return float(nll)

        bounds = [(1e-6, 0.999), (1e-6, 0.999)]
        constraints = (
            {"type": "ineq", "fun": lambda ab: 0.999 - ab[0] - ab[1]},
        )

        x0 = np.asarray(self.dcc_start, dtype=float)
        opt = minimize(
            _negloglike,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if opt.success:
            a_hat, b_hat = map(float, opt.x)
        else:
            a_hat, b_hat = 0.03, 0.95

        q_t = qbar.copy()
        for t in range(1, len(z)):
            z_prev = z[t - 1][:, None]
            q_t = (1.0 - a_hat - b_hat) * qbar + a_hat * (z_prev @ z_prev.T) + b_hat * q_t
            q_t = self._regularize_pd(q_t)

        return a_hat, b_hat, qbar, q_t

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError("arch package not available. Install via: pip install arch") from e

        if "Returns" not in X.columns:
            raise KeyError("X must contain a 'Returns' column.")

        r1 = X["Returns"].astype(float)
        r2 = self._aux_series(X)

        r1_arr, r2_arr = self._clean_pair(r1, r2)

        min_obs = max(100, 10 * (self.p + self.q))
        if len(r1_arr) < min_obs:
            raise ValueError(
                f"Not enough observations to fit DCC-GARCH baseline. "
                f"Need at least {min_obs}, got {len(r1_arr)}."
            )

        am1 = arch_model(
            r1_arr,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
        )
        am2 = arch_model(
            r2_arr,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
        )

        self.res1_ = am1.fit(disp="off")
        self.res2_ = am2.fit(disp="off")

        self._params1 = self.res1_.params.copy()
        self._params2 = self.res2_.params.copy()
        self._train_r1 = r1_arr.copy()
        self._train_r2 = r2_arr.copy()
        self._last_train_return = float(r1.iloc[-1])

        z = np.column_stack(
            [
                np.asarray(self.res1_.std_resid, dtype=float),
                np.asarray(self.res2_.std_resid, dtype=float),
            ]
        )
        z = z[np.isfinite(z).all(axis=1)]

        if len(z) < 20:
            raise ValueError("Insufficient finite standardized residuals for DCC fit.")

        a_hat, b_hat, qbar, q_last = self._fit_dcc(z)
        self._dcc_ab = (a_hat, b_hat)
        self._qbar = qbar
        self._q_last = q_last

        # Unconditional correlation from Qbar
        d_unc = np.sqrt(np.clip(np.diag(qbar), 1e-12, None))
        self._rho_unc = float(qbar[0, 1] / (d_unc[0] * d_unc[1]))

        # Unconditional variances from fitted GARCH parameters
        def _uncond_var(params):
            omega = float(params['omega'])
            a_sum = sum(float(params[k]) for k in params.index if k.startswith('alpha'))
            b_sum = sum(float(params[k]) for k in params.index if k.startswith('beta'))
            denom = 1.0 - a_sum - b_sum
            if denom <= 1e-6:
                return omega / 1e-6
            return omega / denom

        self._var1_unc = _uncond_var(self._params1)
        self._var2_unc = _uncond_var(self._params2)

        return self

    def _sigma_forecasts(
        self,
        all_r: np.ndarray,
        params,
        n_train: int,
        n_pred: int,
    ) -> np.ndarray:
        from arch import arch_model

        fixed = arch_model(
            all_r,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
        ).fix(params)

        fcst = fixed.forecast(
            horizon=self.horizon,
            method="analytic",
            start=n_train,   # first forecast uses all training obs
            reindex=False,
        )

        m = np.asarray(fcst.variance.values, dtype=float)
        if m.shape[0] < n_pred:
            raise RuntimeError(
                f"Expected at least {n_pred} variance rows, got {m.shape[0]}."
            )

        return m[-n_pred:]

    def _dcc_update(self, q_prev: np.ndarray, z_prev: np.ndarray) -> np.ndarray:
        a_hat, b_hat = self._dcc_ab
        z_prev = np.asarray(z_prev, dtype=float)[:, None]
        q_next = (1.0 - a_hat - b_hat) * self._qbar + a_hat * (z_prev @ z_prev.T) + b_hat * q_prev
        return self._regularize_pd(q_next)

    def _future_rho_steps(self, q_state: np.ndarray) -> List[float]:
        """Return per-step signed correlations for the next *horizon* days."""
        a_hat, b_hat = self._dcc_ab
        q_t = q_state.copy()
        rhos: List[float] = []

        for _ in range(self.horizon):
            d = np.sqrt(np.clip(np.diag(q_t), 1e-12, None))
            rho = float(q_t[0, 1] / (d[0] * d[1]))
            rhos.append(float(np.clip(rho, -0.999, 0.999)))

            # Mean-forward DCC projection: E[z*z'] ≈ R_t
            r_t = q_t / np.outer(d, d)
            q_t = (1.0 - a_hat - b_hat) * self._qbar + a_hat * r_t + b_hat * q_t
            q_t = self._regularize_pd(q_t)

        return rhos

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError("arch package not available. Install via: pip install arch") from e

        self._check_is_fitted()

        if "Returns" not in X.columns:
            raise KeyError("X must contain a 'Returns' column.")

        r1_test = X["Returns"].astype(float).fillna(0.0).to_numpy(dtype=float) * self.scale
        r2_test = (
            self._aux_series(X, previous_return=self._last_train_return)
            .astype(float)
            .fillna(0.0)
            .to_numpy(dtype=float)
            * self.scale
        )

        n_pred = len(r1_test)
        if n_pred == 0:
            return np.empty(0, dtype=float)

        all_r1 = np.concatenate([self._train_r1, r1_test])
        all_r2 = np.concatenate([self._train_r2, r2_test])
        n_train = len(self._train_r1)

        var1 = self._sigma_forecasts(all_r1, self._params1, n_train, n_pred)
        var2 = self._sigma_forecasts(all_r2, self._params2, n_train, n_pred)

        fixed1 = arch_model(
            all_r1,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
        ).fix(self._params1)

        fixed2 = arch_model(
            all_r2,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
        ).fix(self._params2)

        # Only use test-period standardized residuals to roll DCC forward
        z1_test = np.nan_to_num(
            np.asarray(fixed1.std_resid, dtype=float)[n_train:],
            nan=0.0,
        )
        z2_test = np.nan_to_num(
            np.asarray(fixed2.std_resid, dtype=float)[n_train:],
            nan=0.0,
        )

        preds = np.empty(n_pred, dtype=float)
        q_curr = self._q_last.copy()
        log_var1_unc = np.log(max(self._var1_unc, 1e-20))
        log_var2_unc = np.log(max(self._var2_unc, 1e-20))

        for i in range(n_pred):
            rhos = self._future_rho_steps(q_curr)
            total = 0.0
            for k in range(self.horizon):
                s1 = var1[i, k]
                s2 = var2[i, k]
                rho_k = rhos[k]
                # Residual market surprise: how much the market deviates from
                # its long-run level *beyond* what the asset already reflects.
                asset_dev = np.log(max(s1, 1e-20)) - log_var1_unc
                mkt_dev = np.log(max(s2, 1e-20)) - log_var2_unc
                residual = np.clip(mkt_dev - asset_dev, -1.0, 1.0)
                total += s1 * np.exp(
                    self.rho_weight * rho_k ** 2 * residual
                )
            preds[i] = total / (self.scale ** 2) / (self.returns_multiplier ** 2)

            z_obs = np.array([z1_test[i], z2_test[i]], dtype=float)
            q_curr = self._dcc_update(q_curr, z_obs)

        return preds

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
        use_market: bool = True,
    ):
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
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