"""Abstract interface for universe constituent management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd


class UniverseProvider(ABC):
    """Abstract interface defining universe constituent resolution."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the universe (e.g. 'nifty50', 'nifty50_dynamic')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_dynamic(self) -> bool:
        """Whether constituent membership varies over time."""
        raise NotImplementedError

    @abstractmethod
    def get_constituents(self, as_of: date | pd.Timestamp) -> list[str]:
        """Return the active universe constituent symbols as of a given date.

        Args:
            as_of: Date or Timestamp to evaluate membership.

        Returns:
            List of constituent ticker symbols active on that date.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_symbols(
        self,
        start_date: date | pd.Timestamp,
        end_date: date | pd.Timestamp | None = None,
    ) -> list[str]:
        """Return the union of all symbols active at any point in the interval.

        Args:
            start_date: Beginning of the date interval.
            end_date: End of the date interval (or None for up to latest).

        Returns:
            Sorted list of unique ticker symbols.
        """
        raise NotImplementedError

    def get_events(
        self,
        start_date: date | pd.Timestamp | None = None,
        end_date: date | pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        """Return reconstitution events that occurred within the date interval.

        Args:
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            List of reconstitution event dictionaries.
        """
        return []
