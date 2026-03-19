import pytest
import pandas as pd
import numpy as np

from nifty_quant.domain.metrics import calculate_metrics, PerformanceMetrics


@pytest.fixture
def sample_returns_and_equity():
    """Create a simple positive returns series for testing"""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Daily returns that average ~0.05% (annualized ~13%)
    returns = pd.Series([0.0005] * 252, index=dates)
    equity = pd.Series(1000 * (1 + returns).cumprod(), index=dates)
    return returns, equity


def test_metrics_calculation(sample_returns_and_equity):
    """Test that metrics are calculated and return valid values"""
    returns, equity = sample_returns_and_equity
    
    metrics = calculate_metrics(returns=returns, equity_curve=equity)
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_return > 0  # All positive returns
    assert metrics.annual_return > 0
    assert np.isclose(metrics.volatility_annual, 0.0, atol=1e-10)  # Constant returns = near-zero vol
    assert np.isinf(metrics.sharpe_ratio) or metrics.sharpe_ratio > 1e10  # Very high Sharpe with near-zero vol
    assert np.isclose(metrics.max_drawdown, 0.0, atol=1e-10)  # No drawdown with positive returns
    assert metrics.sortino_ratio == 0.0 or np.isnan(metrics.sortino_ratio)  # No downside returns
    assert metrics.calmar_ratio is None  # Can't compute with max_drawdown ≈ 0


def test_metrics_with_drawdown():
    """Test metrics calculation with a drawdown scenario"""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Create equity curve with a clear drawdown
    values = [1000] * 30 + [800] * 30 + [1100] * 40  # 20% drawdown then recovery
    equity = pd.Series(values, index=dates)
    returns = equity.pct_change().fillna(0.0)
    
    metrics = calculate_metrics(returns=returns, equity_curve=equity)
    
    # Drawdown from 1000 to 800 is 20%
    assert np.isclose(metrics.max_drawdown, -0.2, atol=1e-10)
    assert np.isclose(metrics.total_return, 0.1, atol=1e-10)  # 1100/1000 - 1
    assert metrics.calmar_ratio is not None
    assert metrics.calmar_ratio > 0  # Positive return / positive drawdown


def test_metrics_empty_series_raises():
    """Test that empty series raise ValueError"""
    empty_series = pd.Series(dtype=float)
    
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_metrics(returns=empty_series, equity_curve=empty_series)


def test_metrics_downside_volatility():
    """Test that Sortino ratio uses only downside returns"""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Mix of positive and negative returns with some variance in downside
    returns_list = [0.01] * 50 + [-0.02, -0.015, -0.025, -0.02] * 12 + [-0.02, -0.018]
    returns = pd.Series(returns_list[:100], index=dates)
    equity = pd.Series(1000 * (1 + returns).cumprod(), index=dates)
    
    metrics = calculate_metrics(returns=returns, equity_curve=equity)
    
    # Sortino should use only the negative returns for downside vol
    assert metrics.downside_volatility_annual > 0
    # Sortino penalizes downside more, so should be less than Sharpe
    # (both can be negative with negative excess returns, so compare if both positive)
    if metrics.sharpe_ratio > 0 and metrics.sortino_ratio > 0:
        assert metrics.sortino_ratio < metrics.sharpe_ratio
