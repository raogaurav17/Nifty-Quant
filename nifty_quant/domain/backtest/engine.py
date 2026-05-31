from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from nifty_quant.domain.models import BacktestResult
from nifty_quant.interfaces.price_repository import PriceRepository
from nifty_quant.interfaces.execution_model import ExecutionModel

# Trading-day constants
_DAYS_PER_YEAR = 252


class BacktestEngine:
    """Pure backtesting engine. 12-1 momentum with inverse-vol sizing."""

    def __init__(
        self,
        price_repo: PriceRepository,
        execution_model: ExecutionModel,
        # --- strategy params (momentum_12_1.yaml) ---
        lookback_days: int = 252,
        skip_recent_days: int = 21,
        top_k: int = 10,
        # --- portfolio params (inverse_vol.yaml) ---
        vol_lookback_days: int = 60,
        max_weight: float = 0.10,
        cash_buffer: float = 0.05,
        target_annual_vol: float = 0.10,
        # --- rebalance cadence ---
        rebalance_every: int = 21,          # ~1 calendar month in trading days
    ) -> None:
        self.price_repo = price_repo
        self.execution_model = execution_model

        # momentum
        self.lookback_days = lookback_days
        self.skip_recent_days = skip_recent_days
        self.top_k = top_k

        # sizing
        self.vol_lookback_days = vol_lookback_days
        self.max_weight = max_weight
        self.cash_buffer = cash_buffer
        self.target_annual_vol = target_annual_vol

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
        """Compute momentum selection, inverse-vol sizing, and constraints."""
        all_dates = daily_returns.index

        min_history = max(self.lookback_days + 1, self.vol_lookback_days + 1)
        rebalance_dates = []
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
            w = self._weights_at(prices=prices, daily_returns=daily_returns, as_of=dt)
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

    def _weights_at(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Compute weights for a single rebalance date."""
        selected = self._momentum_selection(prices=prices, as_of=as_of)
        weights = self._inverse_vol_weights(
            daily_returns=daily_returns,
            symbols=selected,
            as_of=as_of,
        )
        return weights.reindex(prices.columns, fill_value=0.0)

    def _momentum_selection(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> List[str]:
        """Select top_k stocks by 12-1 momentum."""
        price_history = prices.loc[:as_of]

        end_idx   = len(price_history) - 1 - self.skip_recent_days
        start_idx = end_idx - self.lookback_days

        if start_idx < 0:
            return list(prices.columns)

        p_end   = price_history.iloc[end_idx]
        p_start = price_history.iloc[start_idx]

        momentum_scores = (p_end / p_start) - 1.0
        momentum_scores = momentum_scores.dropna()

        top_k = min(self.top_k, len(momentum_scores))
        selected = momentum_scores.nlargest(top_k).index.tolist()

        return selected

    def _inverse_vol_weights(
        self,
        daily_returns: pd.DataFrame,
        symbols: List[str],
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Weight stocks inversely to volatility; apply caps and constraints."""
        ret_slice = daily_returns.loc[:as_of, symbols].tail(self.vol_lookback_days)
        vols = ret_slice.std(ddof=1)
        vols = vols.replace(0.0, np.nan)
        inv_vol = (1.0 / vols).fillna(0.0)

        total = inv_vol.sum()
        if total == 0.0:
            raw_weights = pd.Series(1.0 / len(symbols), index=symbols)
        else:
            raw_weights = inv_vol / total

        weights = self._apply_weight_cap(raw_weights)
        weights = weights * (1.0 - self.cash_buffer)
        weights = self._apply_vol_target(
            weights=weights,
            daily_returns=daily_returns,
            symbols=symbols,
            as_of=as_of,
        )

        return weights

    def _apply_weight_cap(self, weights: pd.Series) -> pd.Series:
        """
        Iteratively redistribute weight from capped stocks to uncapped ones
        until all weights are ≤ max_weight (or all are capped).
        """
        w = weights.copy()
        for _ in range(100):                    # iterate until stable
            over  = w > self.max_weight
            under = ~over

            if not over.any():
                break

            excess = (w[over] - self.max_weight).sum()
            w[over] = self.max_weight

            if under.any() and w[under].sum() > 0:
                w[under] += excess * (w[under] / w[under].sum())
            else:
                break                           # everything is capped

        return w

    def _apply_vol_target(
        self,
        weights: pd.Series,
        daily_returns: pd.DataFrame,
        symbols: List[str],
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Scale weights to match target annual volatility (cap at 1.0)."""
        ret_slice = daily_returns.loc[:as_of, symbols].tail(self.vol_lookback_days)
        cov = ret_slice.cov()

        w_vec = weights.reindex(symbols, fill_value=0.0).values
        port_var = float(w_vec @ cov.values @ w_vec)

        if port_var <= 0.0:
            return weights

        port_daily_vol   = np.sqrt(port_var)
        port_annual_vol  = port_daily_vol * np.sqrt(_DAYS_PER_YEAR)
        target_daily_vol = self.target_annual_vol / np.sqrt(_DAYS_PER_YEAR)

        scalar = min(target_daily_vol / port_daily_vol, 1.0)   # never lever
        return weights * scalar

    # ------------------------------------------------------------------
    # Price alignment
    # ------------------------------------------------------------------

    def _align_prices(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = [df["adj_close"].rename(symbol) for symbol, df in price_data.items()]
        df = pd.concat(aligned, axis=1)

        # Drop symbols missing more than 5% of dates (new listings, delistings)
        min_obs = int(len(df) * 0.95)
        df = df.dropna(axis=1, thresh=min_obs)

        # Forward-fill remaining small gaps (trading halts, holidays)
        df = df.ffill()

        # Drop dates where coverage is too low. For small universes, avoid
        # requiring more symbols than actually exist.
        min_required_symbols = min(self.top_k, len(df.columns))
        df = df.dropna(thresh=min_required_symbols)

        return df