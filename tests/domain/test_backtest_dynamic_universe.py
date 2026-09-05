from datetime import date
import pandas as pd
import pytest

from nifty_quant.domain.backtest.engine import BacktestEngine
from nifty_quant.domain.strategies.base import Strategy
from nifty_quant.domain.universe.dynamic_universe import DynamicUniverseProvider
from tests.domain.fakes import FakePriceRepository, ZeroCostExecutionModel


class EqualWeightEligibleStrategy(Strategy):
    """Simple strategy that equally weights all currently eligible universe stocks."""

    @property
    def min_history_days(self) -> int:
        return 1

    def select_and_weight(
        self,
        prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        cols = list(prices.columns)
        if not cols:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(cols), index=cols)


def test_backtest_engine_rebalances_when_stock_exits():
    # Synthetic timeline:
    # 2020-01-01 to 2020-01-03: AAA and BBB in index
    # 2020-01-03: CCC replaces BBB
    timeline_data = {
        "name": "test_dynamic",
        "baseline_date": "2020-01-05",
        "baseline_constituents": ["AAA", "CCC"],
        "reconstitutions": [
            {
                "effective_date": "2020-01-03",
                "added": ["CCC"],
                "removed": ["BBB"],
            }
        ],
    }
    universe = DynamicUniverseProvider(timeline_data)

    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df_a = pd.DataFrame({"adj_close": [100, 101, 102, 103, 104]}, index=dates)
    df_b = pd.DataFrame({"adj_close": [200, 201, 202, 203, 204]}, index=dates)
    df_c = pd.DataFrame({"adj_close": [50, 51, 52, 53, 54]}, index=dates)

    price_repo = FakePriceRepository({"AAA": df_a, "BBB": df_b, "CCC": df_c})
    execution = ZeroCostExecutionModel()
    strategy = EqualWeightEligibleStrategy()

    engine = BacktestEngine(
        price_repo=price_repo,
        execution_model=execution,
        strategy=strategy,
        rebalance_every=1,  # rebalance every day
    )

    result = engine.run(
        symbols=universe,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 5),
        initial_capital=10_000.0,
    )

    assert result.equity_curve is not None
    assert len(result.equity_curve) > 0

    # Check weights:
    # On 2020-01-02: constituents were AAA, BBB
    # On 2020-01-03: BBB was removed, CCC was added
    w_jan2 = result.weights[pd.Timestamp("2020-01-02")]
    assert w_jan2["AAA"] > 0
    assert w_jan2["BBB"] > 0
    assert w_jan2.get("CCC", 0.0) == 0.0

    w_jan3 = result.weights[pd.Timestamp("2020-01-03")]
    assert w_jan3["AAA"] > 0
    assert w_jan3["CCC"] > 0
    assert w_jan3.get("BBB", 0.0) == 0.0
