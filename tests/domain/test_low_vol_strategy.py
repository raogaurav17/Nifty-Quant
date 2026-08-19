import numpy as np
import pandas as pd
import pytest

from nifty_quant.domain.strategies.low_vol import LowVolatilityStrategy
from nifty_quant.domain.strategies.registry import build_strategy, available_strategies


@pytest.fixture
def sample_market_data():
    """Create sample price and daily return data with varying volatility across stocks."""
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    symbols = ["LOW_VOL", "MED_VOL", "HIGH_VOL"]

    np.random.seed(42)
    ret_low = np.random.normal(0.0005, 0.005, size=300)  # low vol
    ret_med = np.random.normal(0.0005, 0.015, size=300)  # med vol
    ret_high = np.random.normal(0.0005, 0.035, size=300) # high vol

    daily_returns = pd.DataFrame(
        {"LOW_VOL": ret_low, "MED_VOL": ret_med, "HIGH_VOL": ret_high},
        index=dates,
    )
    prices = (1 + daily_returns).cumprod() * 100.0

    return prices, daily_returns, dates[-1]


def test_registry_integration():
    """Test that low_vol is registered and instantiable via registry."""
    assert "low_vol" in available_strategies()

    strat = build_strategy({"name": "low_vol", "lookback_days": 100, "top_k": 2})
    assert isinstance(strat, LowVolatilityStrategy)
    assert strat.lookback_days == 100
    assert strat.top_k == 2


def test_low_vol_selection(sample_market_data):
    """Test that the stock with the lowest volatility is ranked highest/selected first."""
    prices, daily_returns, as_of = sample_market_data

    strat = LowVolatilityStrategy(lookback_days=250, top_k=2)
    selected = strat._low_vol_selection(daily_returns=daily_returns, as_of=as_of)

    assert len(selected) == 2
    assert selected[0] == "LOW_VOL"
    assert selected[1] == "MED_VOL"


def test_compute_signals(sample_market_data):
    """Test that compute_signals returns inverse volatility scores."""
    prices, daily_returns, as_of = sample_market_data

    strat = LowVolatilityStrategy(lookback_days=250)
    signals = strat.compute_signals(prices=prices, daily_returns=daily_returns, as_of=as_of)

    assert "LOW_VOL" in signals
    assert "MED_VOL" in signals
    assert "HIGH_VOL" in signals

    # Higher inverse vol score means lower volatility
    assert signals["LOW_VOL"] > signals["MED_VOL"] > signals["HIGH_VOL"]


def test_select_and_weight(sample_market_data):
    """Test position weights computation and capping."""
    prices, daily_returns, as_of = sample_market_data

    strat = LowVolatilityStrategy(
        lookback_days=250,
        top_k=2,
        max_weight=0.60,
        cash_buffer=0.05,
    )

    weights = strat.select_and_weight(prices=prices, daily_returns=daily_returns, as_of=as_of)

    assert set(weights.index) == set(prices.columns)
    assert weights["HIGH_VOL"] == 0.0  # Not selected
    assert weights["LOW_VOL"] > 0.0
    assert weights["MED_VOL"] > 0.0
    assert weights.max() <= 0.60 + 1e-6
    assert weights.sum() <= (1.0 - 0.05) + 1e-6
