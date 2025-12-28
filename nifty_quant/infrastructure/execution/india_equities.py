from nifty_quant.interfaces.execution_model import ExecutionModel


class IndiaEquitiesExecutionModel(ExecutionModel):
    """
    Simple and realistic execution cost model for Indian equities.

    Costs are applied as:
        cost = notional * turnover * (brokerage + slippage)
    """

    def __init__(
        self,
        brokerage_rate: float = 0.0003,  # 3 bps
        slippage_rate: float = 0.0005,  # 5 bps
    ) -> None:
        self.brokerage_rate = brokerage_rate
        self.slippage_rate = slippage_rate

    def apply_costs(
        self,
        notional: float,
        turnover: float,
    ) -> float:
        """
        Parameters
        ----------
        notional : float
            Total portfolio value
        turnover : float
            Sum of absolute changes in weights

        Returns
        -------
        float
            Total execution cost in currency units
        """
        if turnover <= 0.0:
            return 0.0

        total_rate = self.brokerage_rate + self.slippage_rate
        return notional * turnover * total_rate
