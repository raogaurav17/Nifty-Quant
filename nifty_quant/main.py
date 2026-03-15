import hydra
from datetime import date

from omegaconf import DictConfig, OmegaConf

from nifty_quant.bootstrap.config_schema import AppConfig
from nifty_quant.domain.backtest.engine import BacktestEngine
from nifty_quant.infrastructure.data.yahoo_price_repository import YahooPriceRepository
from nifty_quant.infrastructure.execution.india_equities import IndiaEquitiesExecutionModel

# from nifty_quant.application.run_backtest import run_backtest


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Application entry point.
    - Loads Hydra config
    - Validates it against typed schema
    - Delegates execution to application layer
    """

    # Convert Hydra config to plain dict
    cfg_dict = OmegaConf.to_object(cfg)

    # Typed, validated config (fails fast if invalid)
    _ = AppConfig(**cfg_dict)

    data_cfg = cfg_dict.get("data", {})
    universe_cfg = cfg_dict.get("universe", {})
    provider = str(data_cfg.get("provider", ""))
    symbols = universe_cfg.get("symbols", [])

    if provider != "yahoo":
        raise ValueError(f"Unsupported data provider: {provider}")
    if not symbols:
        raise ValueError("universe.symbols must contain at least one symbol")

    price_repo = YahooPriceRepository()
    execution_cfg = cfg_dict.get("execution", {})
    execution_model = IndiaEquitiesExecutionModel(
        brokerage_rate=float(execution_cfg["brokerage_cost"]),
        slippage_rate=float(execution_cfg["slippage"]),
    )
    engine = BacktestEngine(price_repo=price_repo, execution_model=execution_model)

    backtest_cfg = cfg_dict.get("backtest", {})
    start_date = _parse_date(backtest_cfg["start_date"])
    if start_date is None:
        raise ValueError("backtest.start_date must be set")

    end_date = _parse_date(backtest_cfg.get("end_date"))
    initial_capital = float(backtest_cfg["initial_capital"])

    result = engine.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )

    if result.equity_curve.empty:
        raise RuntimeError("Backtest produced no output. Check symbols/date range/data source.")

    print("Backtest completed")
    print(f"Start equity: {result.equity_curve.iloc[0]:,.2f}")
    print(f"End equity:   {result.equity_curve.iloc[-1]:,.2f}")
    print(f"Total return: {((result.equity_curve.iloc[-1] / initial_capital) - 1.0):.2%}")
    print(f"Observations: {len(result.returns)}")


if __name__ == "__main__":
    main()
