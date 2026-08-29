from __future__ import annotations

from abc import ABC, abstractmethod


class ExecutionModel(ABC):
    """Abstract interface for trade execution and transaction cost modeling."""

    @abstractmethod
    def apply_costs(
        self,
        notional: float,
        turnover: float,
    ) -> float:
        """Calculate the total execution and friction costs for a trade.

        Args:
            notional: Portfolio gross capital / notional value.
            turnover: Portfolio turnover fraction (sum of absolute weight changes).

        Returns:
            Total friction cost in monetary units.
        """
        raise NotImplementedError
