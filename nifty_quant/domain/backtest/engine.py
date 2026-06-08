"""Strategy-agnostic backtesting engine."""

from __future__ import annotations

from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from nifty_quant.domain.models import BacktestResult
from nifty_quant.domain.strategies.base import Strategy
from nifty_quant.domain.strategies.momentum_12_1 import Momentum12_1Strategy
from nifty_quant.interfaces.price_repository import PriceRepository
from nifty_quant.interfaces.execution_model import ExecutionModel


class BacktestEngine:
    """Drives the event loop and delegates signals to the Strategy."""

    def __init__(
        self,
        price_repo: PriceRepository,
        execution_model: ExecutionModel,
        strategy: Strategy | None = None,
        rebalance_every: int = 21,
    ) -> None:
        self.price_repo = price_repo
        self.execution_model = execution_model
        self.strategy = strategy if strategy is not None else Momentum12_1Strategy()
        self.rebalance_every = rebalance_every



    def run(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date | None,
        initial_capital: float,
    ) -> BacktestResult:

        price_data = self.price_repo.get_prices(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        prices = self._align_prices(price_data)

        daily_returns = prices.pct_change().dropna()
        weights = self._build_weights(prices=prices, daily_returns=daily_returns)

        # Apply weights from yesterday to today's return
        portfolio_returns = (weights.shift(1) * daily_returns).sum(axis=1)

        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        costs = pd.Series(0.0, index=portfolio_returns.index)

        for dt in costs.index:
            absolute_cost = self.execution_model.apply_costs(
                notional=initial_capital,
                turnover=turnover.loc[dt],
            )
            costs.loc[dt] = absolute_cost / initial_capital

        net_returns = portfolio_returns - costs
        equity_curve = (1 + net_returns).cumprod() * initial_capital
        trades = turnover.to_frame(name="turnover")

        return BacktestResult(
            equity_curve=equity_curve,
            returns=net_returns,
            weights={dt: weights.loc[dt] for dt in weights.index},
            trades=trades,
        )



    def _build_weights(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Schedule rebalance and build weights."""
        all_dates = daily_returns.index
        min_history = self.strategy.min_history_days

        rebalance_dates: List[pd.Timestamp] = []
        for i, dt in enumerate(all_dates):
            price_loc = prices.index.get_loc(dt)
            if price_loc < min_history:
                continue
            if not rebalance_dates:
                rebalance_dates.append(dt)
            elif (i - all_dates.get_loc(rebalance_dates[-1])) >= self.rebalance_every:
                rebalance_dates.append(dt)

        sparse_weights: Dict[pd.Timestamp, pd.Series] = {}
        for dt in rebalance_dates:
            w = self.strategy.select_and_weight(
                prices=prices,
                daily_returns=daily_returns,
                as_of=dt,
            )
            sparse_weights[dt] = w

        if not sparse_weights:
            n = daily_returns.shape[1]
            return pd.DataFrame(
                1.0 / n,
                index=daily_returns.index,
                columns=daily_returns.columns,
            )

        weight_df = pd.DataFrame(sparse_weights).T
        weight_df = weight_df.reindex(all_dates)
        weight_df = weight_df.ffill()
        weight_df = weight_df.fillna(0.0)

        return weight_df



    def _align_prices(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = [df["adj_close"].rename(symbol) for symbol, df in price_data.items()]
        df = pd.concat(aligned, axis=1)

        # Drop symbols missing >5% data
        min_obs = int(len(df) * 0.95)
        df = df.dropna(axis=1, thresh=min_obs)

        # Forward-fill gaps
        df = df.ffill()

        # Drop dates with low coverage
        top_k = getattr(self.strategy, "top_k", 10)
        min_required_symbols = min(top_k, len(df.columns))
        df = df.dropna(thresh=min_required_symbols)

        return df