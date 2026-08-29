from __future__ import annotations

from datetime import date

import pandas as pd

from nifty_quant.interfaces.execution_model import ExecutionModel
from nifty_quant.interfaces.price_repository import PriceRepository


class FakePriceRepository(PriceRepository):
    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self.data = data

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        return {s: self.data[s] for s in symbols}


class ZeroCostExecutionModel(ExecutionModel):
    def apply_costs(self, notional: float, turnover: float) -> float:
        return 0.0

