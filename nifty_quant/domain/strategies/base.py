"""Abstract base class and shared portfolio sizing routines for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Abstract base class defining the strategy interface and position sizing routines."""

    DAYS_PER_YEAR: int = 252
    signal_label: str = "SIGNAL SCORE"
    rank_note: str = "Ranked by strategy signal score."

    @abstractmethod
    def select_and_weight(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Compute portfolio weights for all universe symbols on a rebalance date.

        Args:
            prices: Historical adjusted close prices matrix (columns = symbols).
            daily_returns: Historical daily percentage returns matrix.
            as_of: Timestamp of the rebalance date.

        Returns:
            Series of portfolio weights indexed by symbol (sum of weights <= 1.0).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def min_history_days(self) -> int:
        """Minimum price bars required before generating the first signal."""
        raise NotImplementedError

    def compute_signals(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """Compute raw selection signal scores for all eligible symbols as of a date.

        Args:
            prices: Historical adjusted close prices matrix.
            daily_returns: Historical daily percentage returns matrix.
            as_of: Timestamp as of which to compute signals.

        Returns:
            Dictionary mapping symbol string to numeric signal score.
        """
        return {}

    # ── Shared Position Sizing & Risk Management Helpers ────────────────────────

    def _inverse_vol_weights(
        self,
        daily_returns: pd.DataFrame,
        symbols: list[str],
        as_of: pd.Timestamp,
        vol_lookback_days: int | None = None,
        max_weight: float | None = None,
        cash_buffer: float | None = None,
        target_annual_vol: float | None = None,
    ) -> pd.Series:
        """Weight selected symbols inversely proportional to their realized volatility."""
        if not symbols:
            return pd.Series(dtype=float)

        lookback = vol_lookback_days if vol_lookback_days is not None else getattr(self, "vol_lookback_days", 60)
        c_buffer = cash_buffer if cash_buffer is not None else getattr(self, "cash_buffer", 0.05)

        ret_slice = daily_returns.loc[:as_of, symbols].tail(lookback)
        vols = ret_slice.std(ddof=1).replace(0.0, np.nan)
        inv_vol = (1.0 / vols).fillna(0.0)

        total = inv_vol.sum()
        if total == 0.0:
            raw_weights = pd.Series(1.0 / len(symbols), index=symbols)
        else:
            raw_weights = inv_vol / total

        weights = self._apply_weight_cap(raw_weights, max_weight=max_weight)
        weights = weights * (1.0 - c_buffer)
        weights = self._apply_vol_target(
            weights=weights,
            daily_returns=daily_returns,
            symbols=symbols,
            as_of=as_of,
            vol_lookback_days=lookback,
            target_annual_vol=target_annual_vol,
        )
        return weights

    def _apply_weight_cap(
        self,
        weights: pd.Series,
        max_weight: float | None = None,
    ) -> pd.Series:
        """Iteratively redistribute excess weight from capped positions."""
        w = weights.copy()
        n = len(w)
        if n <= 1:
            return w

        cap = max_weight if max_weight is not None else getattr(self, "max_weight", 0.20)
        # Prevent degenerate equal-weighting when max_weight <= 1/N
        effective_cap = max(cap, 1.5 / n) if cap <= (1.0 / n) else cap

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
        symbols: list[str],
        as_of: pd.Timestamp,
        vol_lookback_days: int | None = None,
        target_annual_vol: float | None = None,
    ) -> pd.Series:
        """Scale weights to hit target_annual_vol; never lever above 1.0."""
        lookback = vol_lookback_days if vol_lookback_days is not None else getattr(self, "vol_lookback_days", 60)
        target_vol = target_annual_vol if target_annual_vol is not None else getattr(self, "target_annual_vol", 0.10)

        ret_slice = daily_returns.loc[:as_of, symbols].tail(lookback)
        cov = ret_slice.cov()

        w_vec = weights.reindex(symbols, fill_value=0.0).values
        port_var = float(w_vec @ cov.values @ w_vec)

        if port_var <= 0.0:
            return weights

        port_daily_vol = np.sqrt(port_var)
        target_daily_vol = target_vol / np.sqrt(self.DAYS_PER_YEAR)

        scalar = min(target_daily_vol / port_daily_vol, 1.0)
        return weights * scalar


