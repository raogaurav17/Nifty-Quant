"""Execution cost model calibrated for Indian equity markets."""

from __future__ import annotations

from nifty_quant.interfaces.execution_model import ExecutionModel


class IndiaEquitiesExecutionModel(ExecutionModel):
    """Execution cost model for Indian equities incorporating brokerage and slippage."""

    def __init__(
        self,
        brokerage_rate: float = 0.0003,  # 3 bps
        slippage_rate: float = 0.0005,  # 5 bps
    ) -> None:
        self.brokerage_rate = brokerage_rate
        self.slippage_rate = slippage_rate

    @property
    def total_cost_rate(self) -> float:
        """Combined proportional friction rate (brokerage + slippage)."""
        return self.brokerage_rate + self.slippage_rate

    def apply_costs(
        self,
        notional: float,
        turnover: float,
    ) -> float:
        """Calculate the total execution cost for traded turnover.

        Args:
            notional: Portfolio gross capital.
            turnover: Turnover fraction (sum of absolute weight adjustments).

        Returns:
            Monetary friction cost incurred.
        """
        if turnover <= 0.0:
            return 0.0

        return notional * turnover * self.total_cost_rate

