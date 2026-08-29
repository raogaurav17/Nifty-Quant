"""Performance metrics for backtest results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    """Immutable container for comprehensive backtest performance metrics."""

    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: float | None
    volatility_annual: float
    downside_volatility_annual: float


def calculate_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics from backtest returns and equity curve.

    Args:
        returns: Series of daily net returns.
        equity_curve: Series of daily portfolio equity values.
        risk_free_rate: Annualized risk-free rate (e.g., 0.065 for 6.5%).

    Returns:
        PerformanceMetrics instance with all calculated statistics.
    """
    if returns.empty or equity_curve.empty:
        raise ValueError("returns and equity_curve cannot be empty")

    initial_value = float(equity_curve.iloc[0])
    final_value = float(equity_curve.iloc[-1])
    total_return = (final_value / initial_value) - 1.0
    n_trading_days = len(returns)
    n_years = n_trading_days / 252.0
    annual_return = (final_value / initial_value) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    daily_vol = float(returns.std(ddof=1))
    volatility_annual = daily_vol * np.sqrt(252)
    daily_rf = risk_free_rate / 252.0
    excess_returns = returns - daily_rf
    sharpe_ratio = float(excess_returns.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0.0
    cumulative_max = equity_curve.cummax()
    drawdown_series = (equity_curve - cumulative_max) / cumulative_max
    max_drawdown = float(drawdown_series.min())
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_daily_vol = float(downside_returns.std(ddof=1))
    else:
        downside_daily_vol = 0.0

    downside_volatility_annual = downside_daily_vol * np.sqrt(252)

    sortino_ratio = (
        float(excess_returns.mean() / downside_daily_vol * np.sqrt(252))
        if downside_daily_vol > 0
        else 0.0
    )

    if max_drawdown < 0:
        calmar_ratio = annual_return / abs(max_drawdown)
    else:
        calmar_ratio = None

    return PerformanceMetrics(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        volatility_annual=volatility_annual,
        downside_volatility_annual=downside_volatility_annual,
    )

