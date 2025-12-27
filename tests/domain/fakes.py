from datetime import date
from typing import Dict, List

import pandas as pd

from src.interfaces.price_repository import PriceRepository
from src.interfaces.execution_model import ExecutionModel


class FakePriceRepository(PriceRepository):
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data

    def get_prices(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date | None = None,
    ) -> Dict[str, pd.DataFrame]:
        return {s: self.data[s] for s in symbols}


class ZeroCostExecutionModel(ExecutionModel):
    def apply_costs(self, notional: float, turnover: float) -> float:
        return 0.0
