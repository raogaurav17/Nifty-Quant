from nifty_quant.infrastructure.execution.india_equities import (
    IndiaEquitiesExecutionModel,
)


def test_zero_turnover_has_zero_cost():
    model = IndiaEquitiesExecutionModel()
    cost = model.apply_costs(notional=1_000_000, turnover=0.0)
    assert cost == 0.0


def test_positive_turnover_has_positive_cost():
    model = IndiaEquitiesExecutionModel(
        brokerage_rate=0.001,  # 10 bps
        slippage_rate=0.002,  # 20 bps
    )

    cost = model.apply_costs(
        notional=1_000_000,
        turnover=0.5,
    )

    expected = 1_000_000 * 0.5 * 0.003
    assert abs(cost - expected) < 1e-6
