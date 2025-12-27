from dataclasses import dataclass
from datetime import date
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    weights: Dict[date, pd.Series]
    trades: pd.DataFrame
