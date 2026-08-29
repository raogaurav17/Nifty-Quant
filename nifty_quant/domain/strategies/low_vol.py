"""Low-Volatility Anomaly strategy with inverse-volatility sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_quant.domain.strategies.base import Strategy
from nifty_quant.domain.strategies.registry import register


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

    @property
    def min_history_days(self) -> int:
        """Minimum historical price bars required before generating signals."""
        return max(self.lookback_days + 1, self.vol_lookback_days + 1)

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

        vol = ret_slice.std(ddof=1) * np.sqrt(self.DAYS_PER_YEAR)
        vol = vol.replace(0.0, np.nan)
        inv_vol = (1.0 / vol).dropna()
        return inv_vol.to_dict()

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
    ) -> list[str]:
        """Return top-k tickers with the lowest trailing realized volatility."""
        ret_slice = daily_returns.loc[:as_of].tail(self.lookback_days)
        if len(ret_slice) < self.lookback_days // 2:
            return list(daily_returns.columns)

        vol = ret_slice.std(ddof=1) * np.sqrt(self.DAYS_PER_YEAR)
        vol = vol.dropna()

        # Smallest volatility = highest rank for low-vol anomaly strategy
        top_k = min(self.top_k, len(vol))
        return vol.nsmallest(top_k).index.tolist()

