"""Strategy-agnostic backtesting engine."""

from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd

from nifty_quant.domain.models import BacktestResult
from nifty_quant.domain.strategies.base import Strategy
from nifty_quant.domain.strategies.momentum_12_1 import Momentum12_1Strategy
from nifty_quant.interfaces.execution_model import ExecutionModel
from nifty_quant.interfaces.price_repository import PriceRepository
from nifty_quant.interfaces.universe_provider import UniverseProvider


class BacktestEngine:
    """Drives the event loop and delegates signals and allocations to the Strategy."""

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
        symbols: UniverseProvider,
        start_date: date,
        end_date: date | None,
        initial_capital: float,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> BacktestResult:
        """Run backtest simulation over historical data.

        Args:
            symbols: UniverseProvider defining the time-aware constituent universe.
            start_date: Start date for the backtest period.
            end_date: Optional end date for the backtest period.
            initial_capital: Starting portfolio value in currency units.
            progress_callback: Optional callback reporting progress (0.0 to 1.0) and status.

        Returns:
            BacktestResult with equity curve, returns, weights, and trade history.
        """
        universe_provider: UniverseProvider = symbols

        if progress_callback:
            progress_callback(0.20, "Fetching historical market data...")

        query_symbols = universe_provider.get_all_symbols(
            start_date=start_date,
            end_date=end_date,
        )

        price_data = self.price_repo.get_prices(
            symbols=query_symbols,
            start_date=start_date,
            end_date=end_date,
        )

        if progress_callback:
            progress_callback(0.35, "Aligning price matrix and filtering coverage...")

        prices = self._align_prices(price_data, is_dynamic=universe_provider.is_dynamic)
        if universe_provider.is_dynamic:
            daily_returns = prices.pct_change().dropna(how="all")
        else:
            daily_returns = prices.pct_change().dropna()

        if progress_callback:
            progress_callback(0.45, "Calculating strategy signals & allocations...")

        weights = self._build_weights(
            prices=prices,
            daily_returns=daily_returns,
            universe=universe_provider,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(0.80, "Applying trade costs and calculating portfolio equity...")

        # Apply weights from yesterday to today's return
        portfolio_returns = (weights.shift(1) * daily_returns).sum(axis=1)

        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        costs = pd.Series(0.0, index=portfolio_returns.index)

        for dt in costs.index:
            absolute_cost = self.execution_model.apply_costs(
                notional=initial_capital,
                turnover=float(turnover.loc[dt]),
            )
            costs.loc[dt] = absolute_cost / initial_capital

        net_returns = portfolio_returns - costs
        equity_curve = (1 + net_returns).cumprod() * initial_capital
        trades = turnover.to_frame(name="turnover")

        if progress_callback:
            progress_callback(0.90, "Finalizing backtest metrics...")

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
        universe: UniverseProvider,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> pd.DataFrame:
        """Schedule rebalances and compute periodic weight allocations."""
        all_dates = daily_returns.index
        min_history = self.strategy.min_history_days

        rebalance_dates: list[pd.Timestamp] = []
        for i, dt in enumerate(all_dates):
            price_loc = prices.index.get_loc(dt)
            if price_loc < min_history:
                continue
            if not rebalance_dates:
                rebalance_dates.append(dt)
            elif (i - all_dates.get_loc(rebalance_dates[-1])) >= self.rebalance_every:
                rebalance_dates.append(dt)

        sparse_weights: dict[pd.Timestamp, pd.Series] = {}
        total_rebalances = len(rebalance_dates)
        for idx, dt in enumerate(rebalance_dates):
            active_symbols = universe.get_constituents(dt)
            eligible_symbols = [s for s in active_symbols if s in prices.columns]
            if not eligible_symbols:
                eligible_symbols = list(prices.columns)

            prices_sub = prices.loc[:, eligible_symbols]
            returns_sub = daily_returns.loc[:, eligible_symbols]

            w = self.strategy.select_and_weight(
                prices=prices_sub,
                daily_returns=returns_sub,
                as_of=dt,
            )
            sparse_weights[dt] = w
            if progress_callback and total_rebalances > 0:
                pct = 0.45 + 0.30 * ((idx + 1) / total_rebalances)
                date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
                progress_callback(pct, f"Rebalancing strategy ({idx + 1}/{total_rebalances}) - {date_str}")

        if not sparse_weights:
            n = daily_returns.shape[1]
            return pd.DataFrame(
                1.0 / n,
                index=daily_returns.index,
                columns=daily_returns.columns,
            )

        weight_df = pd.DataFrame(sparse_weights).T
        weight_df = weight_df.reindex(columns=prices.columns).fillna(0.0)
        weight_df = weight_df.reindex(all_dates)
        weight_df = weight_df.ffill()
        weight_df = weight_df.fillna(0.0)

        return weight_df

    def _align_prices(
        self,
        price_data: dict[str, pd.DataFrame],
        is_dynamic: bool = False,
    ) -> pd.DataFrame:
        """Extract and align adjusted close prices into a single continuous DataFrame."""
        aligned = [df["adj_close"].rename(symbol) for symbol, df in price_data.items() if "adj_close" in df.columns]
        if not aligned:
            return pd.DataFrame()

        df = pd.concat(aligned, axis=1)

        if is_dynamic:
            # Require at least min_history_days observations, but never more than
            # 50% of total available rows so short backtest windows (e.g. tests)
            # don't drop all columns.
            strategy_min = getattr(self.strategy, "min_history_days", 20)
            min_req = max(1, min(strategy_min, len(df) // 2))
            df = df.dropna(axis=1, thresh=min_req)
            # Forward-fill price gaps
            df = df.ffill()
            # Drop dates where all symbols are NaN
            df = df.dropna(how="all")
        else:
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