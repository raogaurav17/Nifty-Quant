from datetime import date
import pytest
import pandas as pd

from nifty_quant.domain.backtest.engine import BacktestEngine
from nifty_quant.interfaces.execution_model import ExecutionModel
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


class FlatRateExecutionModel(ExecutionModel):
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def apply_costs(self, notional: float, turnover: float) -> float:
        return notional * turnover * self.rate


class AlternatingWeightEngine(BacktestEngine):
    def _equal_weight(self, returns: pd.DataFrame) -> pd.DataFrame:
        # Alternate full allocation between assets to force non-zero turnover.
        weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        first_col = returns.columns[0]
        second_col = returns.columns[1]

        for i, dt in enumerate(returns.index):
            target_col = first_col if i % 2 == 0 else second_col
            weights.loc[dt, target_col] = 1.0

        return weights


def test_execution_costs_are_scaled_to_returns_space(simple_price_data):
    repo = FakePriceRepository(simple_price_data)
    execution = FlatRateExecutionModel(rate=0.01)
    engine = AlternatingWeightEngine(repo, execution)

    result = engine.run(
        symbols=["AAA", "BBB"],
        start_date=date(2020, 1, 1),
        end_date=None,
        initial_capital=1000.0,
    )

    # Net returns should remain in realistic return-space (far above -100%).
    assert (result.returns > -1.0).all()


def test_build_recent_trades_filters_zero_turnover():
    from nifty_quant.application.backtest_runner import _build_recent_trades
    from nifty_quant.domain.models import BacktestResult

    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    trades = pd.DataFrame({"turnover": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=dates)
    result = BacktestResult(
        equity_curve=pd.Series([100.0] * 10, index=dates),
        returns=pd.Series([0.0] * 10, index=dates),
        weights={},
        trades=trades,
    )
    recent = _build_recent_trades(result)
    assert len(recent) == 1
    assert recent[0]["turnover"] == 0.5


def test_weight_cap_preserves_inverse_vol_differentiation():
    from nifty_quant.domain.strategies.arima import ARIMAStrategy

    strat = ARIMAStrategy(top_k=10, max_weight=0.20)
    raw = pd.Series([0.25, 0.15, 0.10, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07])
    capped = strat._apply_weight_cap(raw)

    assert capped.iloc[0] == 0.20
    assert len(set(capped)) > 1  # Weights are not all flattened to the same equal value

