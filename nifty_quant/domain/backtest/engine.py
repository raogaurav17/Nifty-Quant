from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from nifty_quant.domain.models import BacktestResult
from nifty_quant.interfaces.price_repository import PriceRepository
from nifty_quant.interfaces.execution_model import ExecutionModel


class BacktestEngine:
    """
    Pure backtesting engine.
    No I/O, no configs, no side effects.
    """

    def __init__(
        self,
        price_repo: PriceRepository,
        execution_model: ExecutionModel,
    ) -> None:
        self.price_repo = price_repo
        self.execution_model = execution_model

    def run(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date | None,
        initial_capital: float,
    ) -> BacktestResult:
        # -------------------------
        # Load price data
        # -------------------------
        price_data = self.price_repo.get_prices(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

        # Align prices into a single DataFrame
        prices = self._align_prices(price_data)

        # -------------------------
        # Compute returns
        # -------------------------
        daily_returns = prices.pct_change().dropna()

        # -------------------------
        # Generate weights (placeholder)
        # -------------------------
        weights = self._equal_weight(daily_returns)

        # -------------------------
        # Portfolio returns
        # -------------------------
        portfolio_returns = (weights.shift(1) * daily_returns).sum(axis=1)

        # -------------------------
        # Transaction costs
        # -------------------------
        turnover = weights.diff().abs().sum(axis=1)
        costs = portfolio_returns.copy()

        for dt in costs.index:
            costs.loc[dt] = self.execution_model.apply_costs(
                notional=initial_capital,
                turnover=turnover.loc[dt],
            )

        net_returns = portfolio_returns - costs

        # -------------------------
        # Equity curve
        # -------------------------
        equity_curve = (1 + net_returns).cumprod() * initial_capital

        # -------------------------
        # Trades (simplified)
        # -------------------------
        trades = turnover.to_frame(name="turnover")

        return BacktestResult(
            equity_curve=equity_curve,
            returns=net_returns,
            weights={dt: weights.loc[dt] for dt in weights.index},
            trades=trades,
        )

    # -------------------------
    # Helpers (pure)
    # -------------------------

    def _align_prices(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align adjusted close prices across symbols.
        """
        aligned = []

        for symbol, df in price_data.items():
            series = df["adj_close"].rename(symbol)
            aligned.append(series)

        return pd.concat(aligned, axis=1).dropna(how="any")

    def _equal_weight(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder portfolio construction.
        Will be replaced by momentum + inverse vol.
        """
        n = returns.shape[1]
        weights = pd.DataFrame(
            1.0 / n,
            index=returns.index,
            columns=returns.columns,
        )
        return weights
