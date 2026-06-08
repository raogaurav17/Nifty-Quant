"""ARIMA / AR signal strategy with inverse-volatility sizing."""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from nifty_quant.domain.strategies.base import Strategy

_DAYS_PER_YEAR = 252
_LOG = logging.getLogger(__name__)




def _batch_ar_ols(
    history: np.ndarray,   # shape (T, N), columns = symbols
    p: int,
) -> np.ndarray:
    """Fit AR(p) via OLS for all N symbols simultaneously."""
    T, N = history.shape
    n_rows = T - p

    if n_rows < 2:
        return np.zeros(N)

    X_stack = np.empty((N, n_rows, p + 1), dtype=np.float64)
    for k in range(p):
        X_stack[:, :, k] = history[p - k - 1: T - k - 1, :].T
    X_stack[:, :, p] = 1.0  # intercept

    y_stack = history[p:, :].T

    gram  = np.einsum("nti,ntj->nij", X_stack, X_stack)
    cross = np.einsum("nti,nt->ni",  X_stack, y_stack)

    # Solve systems, fallback on singularity
    forecasts = np.zeros(N)
    try:
        coeffs = np.linalg.solve(gram, cross[:, :, np.newaxis]).squeeze(-1)  # (N, p+1)
    except np.linalg.LinAlgError:
        coeffs_list = []
        for i in range(N):
            try:
                c, *_ = np.linalg.lstsq(X_stack[i], y_stack[i], rcond=None)
            except Exception:  # noqa: BLE001
                c = np.zeros(p + 1)
            coeffs_list.append(c)
        coeffs = np.array(coeffs_list)  # (N, p+1)


    x_new = np.empty((N, p + 1))
    for k in range(p):
        x_new[:, k] = history[T - 1 - k, :]   # lag k+1
    x_new[:, p] = 1.0

    forecasts = np.einsum("ni,ni->n", coeffs, x_new)
    return forecasts




def _mle_forecast_one(
    series: np.ndarray,
    order: tuple[int, int, int],
) -> Optional[float]:
    """Fit ARIMA via MLE."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
    except ImportError:
        raise ImportError("statsmodels is required for method='mle'.")

    if len(series) < sum(order) + 10:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            result = ARIMA(series, order=order).fit()
            return float(result.forecast(steps=1)[0])
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("ARIMA MLE fit failed: %s", exc)
        return None




class ARIMAStrategy(Strategy):
    """AR / ARIMA signal strategy with inverse-vol position sizing."""

    def __init__(
        self,
        method: str = "ols",
        arima_p: int = 2,
        arima_d: int = 0,
        arima_q: int = 0,
        fit_window: int = 60,
        top_k: int = 10,
        vol_lookback_days: int = 60,
        max_weight: float = 0.10,
        cash_buffer: float = 0.05,
        target_annual_vol: float = 0.10,
        max_workers: int = 4,
    ) -> None:
        if method not in ("ols", "mle"):
            raise ValueError(f"method must be 'ols' or 'mle', got '{method}'")
        self.method = method
        self.p = arima_p
        self.d = arima_d
        self.q = arima_q
        self.order = (arima_p, arima_d, arima_q)
        self.fit_window = fit_window
        self.top_k = top_k
        self.vol_lookback_days = vol_lookback_days
        self.max_weight = max_weight
        self.cash_buffer = cash_buffer
        self.target_annual_vol = target_annual_vol
        self.max_workers = max_workers



    @property
    def min_history_days(self) -> int:
        return max(self.fit_window + 1, self.vol_lookback_days + 1)

    def select_and_weight(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        log_returns = np.log1p(daily_returns.loc[:as_of].tail(self.fit_window))

        if self.method == "ols":
            forecasts_arr = self._ols_forecasts(log_returns)
            symbols = log_returns.columns.tolist()
            forecasts: Dict[str, float] = {
                sym: float(fc) for sym, fc in zip(symbols, forecasts_arr)
            }
        else:
            forecasts = self._mle_forecasts(log_returns)

        selected = self._select_symbols(forecasts)

        if not selected:
            return pd.Series(0.0, index=prices.columns)

        weights = self._inverse_vol_weights(
            daily_returns=daily_returns,
            symbols=selected,
            as_of=as_of,
        )
        return weights.reindex(prices.columns, fill_value=0.0)



    def _ols_forecasts(self, log_returns: pd.DataFrame) -> np.ndarray:
        """Batched AR(p) OLS across all symbols."""
        # Drop columns that are all-NaN; fill remaining NaNs with 0
        valid = log_returns.dropna(axis=1, how="all")
        data = valid.fillna(0.0).values  # (T, N)

        raw = _batch_ar_ols(data, self.p)

        # Map back; symbols not in 'valid' get forecast 0
        n_all = len(log_returns.columns)
        result = np.zeros(n_all)
        valid_idx = [log_returns.columns.get_loc(c) for c in valid.columns]
        result[valid_idx] = raw
        return result



    def _mle_forecasts(self, log_returns: pd.DataFrame) -> Dict[str, float]:
        symbols = log_returns.columns.tolist()

        def _fit_one(sym: str) -> tuple[str, Optional[float]]:
            series = log_returns[sym].dropna().values
            return sym, _mle_forecast_one(series, self.order)

        forecasts: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for sym, fc in pool.map(_fit_one, symbols):
                if fc is not None:
                    forecasts[sym] = fc
        return forecasts



    def _select_symbols(self, forecasts: Dict[str, float]) -> List[str]:
        positive = {sym: fc for sym, fc in forecasts.items() if fc > 0.0}
        if not positive:
            return []
        ranked = sorted(positive, key=positive.__getitem__, reverse=True)
        return ranked[: self.top_k]



    def _inverse_vol_weights(
        self,
        daily_returns: pd.DataFrame,
        symbols: List[str],
        as_of: pd.Timestamp,
    ) -> pd.Series:
        ret_slice = daily_returns.loc[:as_of, symbols].tail(self.vol_lookback_days)
        vols = ret_slice.std(ddof=1).replace(0.0, np.nan)
        inv_vol = (1.0 / vols).fillna(0.0)

        total = inv_vol.sum()
        raw_weights = (
            pd.Series(1.0 / len(symbols), index=symbols)
            if total == 0.0
            else inv_vol / total
        )

        w = self._apply_weight_cap(raw_weights)
        w = w * (1.0 - self.cash_buffer)
        return self._apply_vol_target(w, daily_returns, symbols, as_of)

    def _apply_weight_cap(self, weights: pd.Series) -> pd.Series:
        w = weights.copy()
        for _ in range(100):
            over, under = w > self.max_weight, ~(w > self.max_weight)
            if not over.any():
                break
            excess = (w[over] - self.max_weight).sum()
            w[over] = self.max_weight
            if under.any() and w[under].sum() > 0:
                w[under] += excess * (w[under] / w[under].sum())
            else:
                break
        return w

    def _apply_vol_target(
        self,
        weights: pd.Series,
        daily_returns: pd.DataFrame,
        symbols: List[str],
        as_of: pd.Timestamp,
    ) -> pd.Series:
        ret_slice = daily_returns.loc[:as_of, symbols].tail(self.vol_lookback_days)
        cov = ret_slice.cov()
        w_vec = weights.reindex(symbols, fill_value=0.0).values
        port_var = float(w_vec @ cov.values @ w_vec)
        if port_var <= 0.0:
            return weights
        target_daily_vol = self.target_annual_vol / np.sqrt(_DAYS_PER_YEAR)
        scalar = min(target_daily_vol / np.sqrt(port_var), 1.0)
        return weights * scalar
