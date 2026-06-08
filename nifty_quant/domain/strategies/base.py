"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class Strategy(ABC):
    """Strategy interface."""

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
