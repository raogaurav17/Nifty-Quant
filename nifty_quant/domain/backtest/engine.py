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
    """
    Pure backtesting engine.
    No I/O, no configs, no side effects.

    Strategy implemented:
      - Signal : 12-1 momentum (lookback_days trailing return, skipping
                 the most recent skip_recent_days to avoid short-term reversal).
      - Selection: top_k stocks by momentum score each rebalance date.
      - Sizing  : inverse-volatility weights (vol_lookback_days rolling std),
                  capped at max_weight per stock, scaled by (1 - cash_buffer),
                  and optionally scaled to hit target_annual_vol.
      - Rebalance: monthly (caller controls rebalance dates via `rebalance_every`).
    """

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

        # cadence
        self.rebalance_every = rebalance_every

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date | None,
        initial_capital: float,
    ) -> BacktestResult:

        # ── 1. Load & align prices ──────────────────────────────────────
        price_data = self.price_repo.get_prices(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        prices = self._align_prices(price_data)

        # ── 2. Daily returns ────────────────────────────────────────────
        daily_returns = prices.pct_change().dropna()

        # ── 3. Build weight grid ────────────────────────────────────────
        # Weights are computed at each rebalance date, then forward-filled
        # to every trading day so the portfolio return formula is unchanged.
        weights = self._build_weights(prices=prices, daily_returns=daily_returns)

        # ── 4. Portfolio gross returns ──────────────────────────────────
        # weights.shift(1): yesterday's close weights applied to today's return
        portfolio_returns = (weights.shift(1) * daily_returns).sum(axis=1)

        # ── 5. Transaction costs ────────────────────────────────────────
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        costs = pd.Series(0.0, index=portfolio_returns.index)

        for dt in costs.index:
            absolute_cost = self.execution_model.apply_costs(
                notional=initial_capital,
                turnover=turnover.loc[dt],
            )
            costs.loc[dt] = absolute_cost / initial_capital

        net_returns = portfolio_returns - costs

        # ── 6. Equity curve ─────────────────────────────────────────────
        equity_curve = (1 + net_returns).cumprod() * initial_capital

        # ── 7. Trades ledger ────────────────────────────────────────────
        trades = turnover.to_frame(name="turnover")

        return BacktestResult(
            equity_curve=equity_curve,
            returns=net_returns,
            weights={dt: weights.loc[dt] for dt in weights.index},
            trades=trades,
        )

    # ------------------------------------------------------------------
    # Weight construction
    # ------------------------------------------------------------------

    def _build_weights(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each rebalance date:
          1. Compute 12-1 momentum scores → select top_k.
          2. Size selected stocks by inverse volatility.
          3. Apply max_weight cap, cash_buffer, and vol targeting.

        Returns a daily weight DataFrame (forward-filled between rebalances).
        The weight DataFrame is aligned to daily_returns.index.
        """
        all_dates = daily_returns.index

        # We need at least lookback_days of price history before any signal
        min_history = max(self.lookback_days + 1, self.vol_lookback_days + 1)

        # Identify rebalance dates (every N trading days, starting once
        # enough history exists)
        rebalance_dates = []
        for i, dt in enumerate(all_dates):
            price_loc = prices.index.get_loc(dt)
            if price_loc < min_history:
                continue
            if not rebalance_dates:
                rebalance_dates.append(dt)
            elif (i - all_dates.get_loc(rebalance_dates[-1])) >= self.rebalance_every:
                rebalance_dates.append(dt)


        # Compute weights at each rebalance date
        sparse_weights: Dict[pd.Timestamp, pd.Series] = {}
        for dt in rebalance_dates:
            w = self._weights_at(prices=prices, daily_returns=daily_returns, as_of=dt)
            sparse_weights[dt] = w

        if not sparse_weights:
            # Not enough history — fall back to equal weight (edge case)
            n = daily_returns.shape[1]
            return pd.DataFrame(
                1.0 / n,
                index=daily_returns.index,
                columns=daily_returns.columns,
            )


        # Expand sparse weights to a daily grid via forward-fill
        weight_df = pd.DataFrame(sparse_weights).T          # rebalance_dates × symbols
        weight_df = weight_df.reindex(all_dates)             # align to all trading days
        weight_df = weight_df.ffill()                        # carry last known weights
        weight_df = weight_df.fillna(0.0)                    # pre-signal days → cash

        return weight_df

    def _weights_at(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """
        Compute the full target weight vector for a single rebalance date.
        """
        selected = self._momentum_selection(prices=prices, as_of=as_of)
        weights = self._inverse_vol_weights(
            daily_returns=daily_returns,
            symbols=selected,
            as_of=as_of,
        )
        return weights.reindex(prices.columns, fill_value=0.0)

    # ------------------------------------------------------------------
    # Momentum signal
    # ------------------------------------------------------------------

    def _momentum_selection(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> List[str]:
        """
        12-1 momentum: total return from (as_of - lookback_days) to
        (as_of - skip_recent_days).  Returns top_k symbols by score.
        """
        price_history = prices.loc[:as_of]

        end_idx   = len(price_history) - 1 - self.skip_recent_days
        start_idx = end_idx - self.lookback_days

        if start_idx < 0:
            # Insufficient history — return all symbols (shouldn't happen
            # given the min_history guard in _build_weights)
            return list(prices.columns)

        p_end   = price_history.iloc[end_idx]
        p_start = price_history.iloc[start_idx]

        momentum_scores = (p_end / p_start) - 1.0
        momentum_scores = momentum_scores.dropna()

        top_k = min(self.top_k, len(momentum_scores))
        selected = momentum_scores.nlargest(top_k).index.tolist()

        return selected

    # ------------------------------------------------------------------
    # Inverse-volatility sizing
    # ------------------------------------------------------------------

    def _inverse_vol_weights(
        self,
        daily_returns: pd.DataFrame,
        symbols: List[str],
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """
        1. Compute rolling vol for each selected symbol (vol_lookback_days).
        2. Weight = 1 / vol; normalise to sum = 1.
        3. Clip at max_weight; renormalise.
        4. Scale by (1 - cash_buffer).
        5. Optionally rescale to hit target_annual_vol.
        """
        ret_slice = daily_returns.loc[:as_of, symbols].tail(self.vol_lookback_days)

        vols = ret_slice.std(ddof=1)           # daily vol per symbol
        vols = vols.replace(0.0, np.nan)       # avoid division by zero

        # Symbols with no valid vol get zero weight
        inv_vol = (1.0 / vols).fillna(0.0)

        total = inv_vol.sum()
        if total == 0.0:
            # All vols missing — equal weight the selection
            raw_weights = pd.Series(1.0 / len(symbols), index=symbols)
        else:
            raw_weights = inv_vol / total

        # ── Max-weight cap (iterative renormalisation) ──────────────────
        weights = self._apply_weight_cap(raw_weights)

        # ── Cash buffer ─────────────────────────────────────────────────
        weights = weights * (1.0 - self.cash_buffer)

        # ── Vol targeting ───────────────────────────────────────────────
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
        """
        Scale weights so that the expected portfolio daily vol equals
        target_annual_vol / sqrt(252).  Cap the scalar at 1.0 (long-only:
        we never lever up, only de-lever).
        """
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

        # Drop any dates where fewer than top_k symbols have data
        # (start of history before most symbols existed)
        df = df.dropna(thresh=self.top_k)

        return df