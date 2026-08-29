from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class PriceRepository(ABC):
    """Abstract interface for historical price data access."""

    @abstractmethod
    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical price data for the specified symbols.

        Args:
            symbols: List of ticker symbols to query.
            start_date: Starting date for price history (inclusive).
            end_date: Optional ending date for price history (inclusive).

        Returns:
            Dictionary mapping symbol string to its historical DataFrame.
        """
        raise NotImplementedError
