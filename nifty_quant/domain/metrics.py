"""
Performance metrics for backtest results.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    """Container for backtest performance metrics."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: Optional[float]
    volatility_annual: float
    downside_volatility_annual: float


def calculate_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest results.

    Parameters
    ----------
    returns : pd.Series
        Net daily returns (with costs already deducted)
    equity_curve : pd.Series
        Daily equity curve (portfolio value over time)
    risk_free_rate : float
        Annual risk-free rate (default 0.0)

    Returns
    -------
    PerformanceMetrics
        Dataclass containing all performance metrics
    """
    if returns.empty or equity_curve.empty:
        raise ValueError("returns and equity_curve cannot be empty")

    # ── Total & Annual Return ───────────────────────────────────────────
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]
    total_return = (final_value / initial_value) - 1.0

    # Annualized return: assume 252 trading days/year
    n_trading_days = len(returns)
    n_years = n_trading_days / 252.0
    annual_return = (final_value / initial_value) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    # ── Volatility ──────────────────────────────────────────────────────
    # Daily volatility
    daily_vol = returns.std(ddof=1)
    volatility_annual = daily_vol * np.sqrt(252)

    # ── Sharpe Ratio ────────────────────────────────────────────────────
    # Excess return per unit of risk
    daily_rf = risk_free_rate / 252.0
    excess_returns = returns - daily_rf
    sharpe_ratio = (excess_returns.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0.0

    # ── Max Drawdown ────────────────────────────────────────────────────
    # Peak-to-trough decline from any historical peak
    cumulative_max = equity_curve.cummax()
    drawdown_series = (equity_curve - cumulative_max) / cumulative_max
    max_drawdown = drawdown_series.min()  # Will be negative

    # ── Downside Volatility & Sortino Ratio ────────────────────────────
    # Only negative returns (downside risk)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_daily_vol = downside_returns.std(ddof=1)
    else:
        downside_daily_vol = 0.0

    downside_volatility_annual = downside_daily_vol * np.sqrt(252)

    sortino_ratio = (
        (excess_returns.mean() / downside_daily_vol * np.sqrt(252))
        if downside_daily_vol > 0
        else 0.0
    )

    # ── Calmar Ratio ───────────────────────────────────────────────────
    # Annual return divided by absolute max drawdown
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
