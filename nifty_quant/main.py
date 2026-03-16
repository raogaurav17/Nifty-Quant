import hydra
from datetime import date

from omegaconf import DictConfig, OmegaConf

from nifty_quant.bootstrap.config_schema import AppConfig
from nifty_quant.domain.backtest.engine import BacktestEngine
from nifty_quant.infrastructure.data.yahoo_price_repository import YahooPriceRepository
from nifty_quant.infrastructure.execution.india_equities import IndiaEquitiesExecutionModel

from dateutil.relativedelta import relativedelta

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
    strategy_cfg  = cfg_dict.get("strategy", {})
    portfolio_cfg = cfg_dict.get("portfolio", {})

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

    

    engine = BacktestEngine(
        price_repo=price_repo,
        execution_model=execution_model,
        lookback_days=int(strategy_cfg.get("lookback_days", 252)),
        skip_recent_days=int(strategy_cfg.get("skip_recent_days", 21)),
        top_k=int(strategy_cfg.get("top_k", 10)),
        vol_lookback_days=int(portfolio_cfg.get("vol_lookback_days", 60)),
        max_weight=float(portfolio_cfg.get("max_weight", 0.10)),
        cash_buffer=float(portfolio_cfg.get("cash_buffer", 0.05)),
        target_annual_vol=float(portfolio_cfg.get("target_annual_vol", 0.10)),
    )

    backtest_cfg = cfg_dict.get("backtest", {})
    start_date = _parse_date(backtest_cfg["start_date"])
    if start_date is None:
        raise ValueError("backtest.start_date must be set")

    end_date = _parse_date(backtest_cfg.get("end_date"))
    initial_capital = float(backtest_cfg["initial_capital"])

    # Pull 14 months extra so momentum signal has warmup data
    fetch_start = start_date - relativedelta(months=14)

    result = engine.run(
        symbols=symbols,
        start_date=fetch_start,
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
