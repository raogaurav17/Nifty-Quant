"""Domain data models for backtest outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Immutable domain model representing the complete output of a backtest run."""

    equity_curve: pd.Series
    returns: pd.Series
    weights: dict[date, pd.Series]
    trades: pd.DataFrame

