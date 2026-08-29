"""Momentum 12-1 strategy with inverse-volatility sizing."""

from __future__ import annotations

import pandas as pd

from nifty_quant.domain.strategies.base import Strategy
from nifty_quant.domain.strategies.registry import register


@register("momentum_12_1")
class Momentum12_1Strategy(Strategy):
    """Cross-sectional 12-1 momentum strategy with inverse-volatility position sizing."""

    signal_label: str = "MOM SCORE"
    rank_note: str = "Ranked by 12-1 momentum score -- the actual selection signal."

    def __init__(
        self,
        lookback_days: int = 252,
        skip_recent_days: int = 21,
        top_k: int = 10,
        vol_lookback_days: int = 60,
        max_weight: float = 0.20,
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
        """Minimum historical price bars required before generating signals."""
        return max(self.lookback_days + 1, self.vol_lookback_days + 1)

    def compute_signals(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """Compute 12-1 momentum scores as of a given timestamp."""
        price_history = prices.loc[:as_of]
        end_idx = len(price_history) - 1 - self.skip_recent_days
        start_idx = end_idx - self.lookback_days

        if start_idx < 0:
            return {}

        p_end = price_history.iloc[end_idx]
        p_start = price_history.iloc[start_idx]

        momentum_scores = (p_end / p_start) - 1.0
        return momentum_scores.dropna().to_dict()

    def select_and_weight(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Select top-k momentum symbols and compute inverse-volatility weights."""
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
    ) -> list[str]:
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

