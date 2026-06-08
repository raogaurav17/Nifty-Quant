from nifty_quant.interfaces.execution_model import ExecutionModel


class IndiaEquitiesExecutionModel(ExecutionModel):
    """Execution cost model for Indian equities."""

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
        
        if turnover <= 0.0:
            return 0.0

        total_rate = self.brokerage_rate + self.slippage_rate
        return notional * turnover * total_rate
