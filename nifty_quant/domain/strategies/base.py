"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class Strategy(ABC):
    """Strategy interface."""

    signal_label: str = "SIGNAL SCORE"
    rank_note: str = "Ranked by strategy signal score."

    @abstractmethod
    def select_and_weight(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Compute portfolio weights for a single rebalance date."""

    @property
    @abstractmethod
    def min_history_days(self) -> int:
        """Minimum price bars required before first signal."""

    def compute_signals(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """Compute raw selection signal scores for all symbols as of a date."""
        return {}

