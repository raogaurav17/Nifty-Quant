from datetime import date
import pytest
import pandas as pd

from nifty_quant.domain.backtest.engine import BacktestEngine
from tests.domain.fakes import FakePriceRepository, ZeroCostExecutionModel


@pytest.fixture
def simple_price_data():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    df1 = pd.DataFrame(
        {
            "adj_close": [100, 101, 102, 103, 104],
            "volume": [1000] * 5,
        },
        index=dates,
    )

    df2 = pd.DataFrame(
        {
            "adj_close": [200, 202, 204, 206, 208],
            "volume": [1000] * 5,
        },
        index=dates,
    )
    return {
        "AAA": df1,
        "BBB": df2,
    }


def test_backtest_runs_and_produces_equity_curve(simple_price_data):
    repo = FakePriceRepository(simple_price_data)
    execution = ZeroCostExecutionModel()

    engine = BacktestEngine(
        price_repo=repo,
        execution_model=execution,
    )

    result = engine.run(
        symbols=["AAA", "BBB"],
        start_date=date(2020, 1, 1),
        end_date=None,
        initial_capital=1000.0,
    )

    # --- Assertions ---
    assert result.equity_curve is not None
    assert isinstance(result.equity_curve, pd.Series)
    assert len(result.equity_curve) > 0

    # Capital should grow (prices monotonically increase)
    assert result.equity_curve.iloc[-1] > 1000.0


def test_portfolio_weights_sum_to_one(simple_price_data):
    repo = FakePriceRepository(simple_price_data)
    execution = ZeroCostExecutionModel()

    engine = BacktestEngine(repo, execution)

    result = engine.run(
        symbols=["AAA", "BBB"],
        start_date=date(2020, 1, 1),
        end_date=None,
        initial_capital=1000.0,
    )

    for dt, weights in result.weights.items():
        assert abs(weights.sum() - 1.0) < 1e-8


def test_zero_execution_costs(simple_price_data):
    repo = FakePriceRepository(simple_price_data)
    execution = ZeroCostExecutionModel()

    engine = BacktestEngine(repo, execution)

    result = engine.run(
        symbols=["AAA", "BBB"],
        start_date=date(2020, 1, 1),
        end_date=None,
        initial_capital=1000.0,
    )

    # No NaNs or weird behavior
    assert result.returns.isnull().sum() == 0
