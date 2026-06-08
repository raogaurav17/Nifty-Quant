"""Momentum 12-1 strategy with inverse-volatility sizing."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from nifty_quant.domain.strategies.base import Strategy

# Trading-day constant
_DAYS_PER_YEAR = 252


class Momentum12_1Strategy(Strategy):
    """Cross-sectional 12-1 momentum with inverse-vol position sizing."""

    def __init__(
        self,
        lookback_days: int = 252,
        skip_recent_days: int = 21,
        top_k: int = 10,
        vol_lookback_days: int = 60,
        max_weight: float = 0.10,
        cash_buffer: float = 0.05,
        target_annual_vol: float = 0.10,
    ) -> None:
        self.lookback_days = lookback_days
        self.skip_recent_days = skip_recent_days
        self.top_k = top_k
        self.vol_lookback_days = vol_lookback_days
        self.max_weight = max_weight
        self.cash_buffer = cash_buffer
        self.target_annual_vol = target_annual_vol



    @property
    def min_history_days(self) -> int:
        return max(self.lookback_days + 1, self.vol_lookback_days + 1)

    def select_and_weight(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        selected = self._momentum_selection(prices=prices, as_of=as_of)
        weights = self._inverse_vol_weights(
            daily_returns=daily_returns,
            symbols=selected,
            as_of=as_of,
        )
        return weights.reindex(prices.columns, fill_value=0.0)



    def _momentum_selection(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> List[str]:
        """Return the top-k tickers ranked by 12-1 momentum."""
        price_history = prices.loc[:as_of]

        end_idx = len(price_history) - 1 - self.skip_recent_days
        start_idx = end_idx - self.lookback_days

        if start_idx < 0:
            return list(prices.columns)

        p_end = price_history.iloc[end_idx]
        p_start = price_history.iloc[start_idx]

        momentum_scores = (p_end / p_start) - 1.0
        momentum_scores = momentum_scores.dropna()

        top_k = min(self.top_k, len(momentum_scores))
        return momentum_scores.nlargest(top_k).index.tolist()

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
        for _ in range(100):
            over = w > self.max_weight
            under = ~over
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
