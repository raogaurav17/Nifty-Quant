"""Low-Volatility Anomaly strategy with inverse-volatility sizing."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from nifty_quant.domain.strategies.base import Strategy
from nifty_quant.domain.strategies.registry import register

_DAYS_PER_YEAR = 252


@register("low_vol")
class LowVolatilityStrategy(Strategy):
    """Selects stocks with the lowest realised volatility over a lookback window."""

    signal_label: str = "INV VOL SCORE"
    rank_note: str = "Ranked by inverse realised volatility score (higher = lower risk)."

    def __init__(
        self,
        lookback_days: int = 252,
        top_k: int = 10,
        vol_lookback_days: int = 60,
        max_weight: float = 0.20,
        cash_buffer: float = 0.05,
        target_annual_vol: float = 0.10,
    ) -> None:
        self.lookback_days = lookback_days
        self.top_k = top_k
        self.vol_lookback_days = vol_lookback_days
        self.max_weight = max_weight
        self.cash_buffer = cash_buffer
        self.target_annual_vol = target_annual_vol

    def compute_signals(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """Return inverse annualized volatility scores for symbols up to as_of."""
        ret_slice = daily_returns.loc[:as_of].tail(self.lookback_days)
        if len(ret_slice) < self.lookback_days // 2:
            return {}

        vol = ret_slice.std(ddof=1) * np.sqrt(_DAYS_PER_YEAR)
        vol = vol.replace(0.0, np.nan)
        inv_vol = (1.0 / vol).dropna()
        return inv_vol.to_dict()

    @property
    def min_history_days(self) -> int:
        return max(self.lookback_days + 1, self.vol_lookback_days + 1)

    def select_and_weight(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Select top_k lowest volatility stocks and compute portfolio weights."""
        selected = self._low_vol_selection(daily_returns=daily_returns, as_of=as_of)
        weights = self._inverse_vol_weights(
            daily_returns=daily_returns,
            symbols=selected,
            as_of=as_of,
        )
        return weights.reindex(prices.columns, fill_value=0.0)

    def _low_vol_selection(
        self,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> List[str]:
        """Return top-k tickers with the lowest trailing realized volatility."""
        ret_slice = daily_returns.loc[:as_of].tail(self.lookback_days)
        if len(ret_slice) < self.lookback_days // 2:
            return list(daily_returns.columns)

        vol = ret_slice.std(ddof=1) * np.sqrt(_DAYS_PER_YEAR)
        vol = vol.dropna()

        # Smallest volatility = highest rank for low-vol anomaly strategy
        top_k = min(self.top_k, len(vol))
        return vol.nsmallest(top_k).index.tolist()

    def _inverse_vol_weights(
        self,
        daily_returns: pd.DataFrame,
        symbols: List[str],
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Weight selected symbols inversely to their realised volatility."""
        ret_slice = daily_returns.loc[:as_of, symbols].tail(self.vol_lookback_days)
        vols = ret_slice.std(ddof=1)
        vols = vols.replace(0.0, np.nan)
        inv_vol = (1.0 / vols).fillna(0.0)

        total = inv_vol.sum()
        if total == 0.0:
            raw_weights = pd.Series(1.0 / len(symbols), index=symbols)
        else:
            raw_weights = inv_vol / total

        weights = self._apply_weight_cap(raw_weights)
        weights = weights * (1.0 - self.cash_buffer)
        weights = self._apply_vol_target(
            weights=weights,
            daily_returns=daily_returns,
            symbols=symbols,
            as_of=as_of,
        )
        return weights

    def _apply_weight_cap(self, weights: pd.Series) -> pd.Series:
        """Iteratively redistribute excess weight from capped positions."""
        w = weights.copy()
        n = len(w)
        if n <= 1:
            return w

        effective_cap = max(self.max_weight, 1.5 / n) if self.max_weight <= (1.0 / n) else self.max_weight

        for _ in range(100):
            over = w > effective_cap
            under = ~over
            if not over.any():
                break
            excess = (w[over] - effective_cap).sum()
            w[over] = effective_cap
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
        """Scale weights to hit target_annual_vol; never lever above 1.0."""
        ret_slice = daily_returns.loc[:as_of, symbols].tail(self.vol_lookback_days)
        cov = ret_slice.cov()

        w_vec = weights.reindex(symbols, fill_value=0.0).values
        port_var = float(w_vec @ cov.values @ w_vec)

        if port_var <= 0.0:
            return weights

        port_daily_vol = np.sqrt(port_var)
        target_daily_vol = self.target_annual_vol / np.sqrt(_DAYS_PER_YEAR)

        scalar = min(target_daily_vol / port_daily_vol, 1.0)
        return weights * scalar
