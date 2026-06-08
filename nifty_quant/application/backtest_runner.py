from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from nifty_quant.domain.backtest.engine import BacktestEngine
from nifty_quant.domain.strategies.registry import build_strategy
from nifty_quant.domain.metrics import PerformanceMetrics, calculate_metrics
from nifty_quant.domain.models import BacktestResult
from nifty_quant.infrastructure.data.yahoo_price_repository import YahooPriceRepository
from nifty_quant.infrastructure.execution.india_equities import IndiaEquitiesExecutionModel

from dateutil.relativedelta import relativedelta


@dataclass(frozen=True)
class BacktestSnapshot:
    config: dict[str, Any]
    result: BacktestResult
    metrics: PerformanceMetrics
    chart_path: str
    chart_min: float
    chart_max: float
    holdings: list[dict[str, Any]]
    recent_trades: list[dict[str, Any]]
    start_date: date
    end_date: date | None
    initial_capital: float
    symbols: list[str]


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(str(value))


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load and compose the Hydra config programmatically."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from pathlib import Path

    config_dir = str(Path(__file__).resolve().parents[2] / "conf")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


def _build_chart_path(equity_curve: pd.Series) -> tuple[str, float, float]:
    if equity_curve.empty:
        return "", 0.0, 0.0

    sampled = equity_curve.iloc[:: max(len(equity_curve) // 120, 1)]
    if sampled.index[-1] != equity_curve.index[-1]:
        sampled = pd.concat([sampled, equity_curve.iloc[[-1]]])

    width = 1200.0
    height = 360.0
    padding = 28.0
    values = sampled.astype(float).tolist()
    low = min(values)
    high = max(values)
    span = max(high - low, 1e-9)
    step_x = (width - 2 * padding) / max(len(values) - 1, 1)

    points: list[str] = []
    for index, value in enumerate(values):
        x = padding + step_x * index
        normalized = (value - low) / span
        y = height - padding - normalized * (height - 2 * padding)
        points.append(f"{x:.1f},{y:.1f}")

    return f"M {points[0]}" + " " + " ".join(f"L {point}" for point in points[1:]), low, high


def _build_holdings(result: BacktestResult) -> list[dict[str, Any]]:
    if not result.weights:
        return []

    latest_date = max(result.weights)
    latest_weights = result.weights[latest_date].sort_values(ascending=False)
    holdings = []
    for symbol, weight in latest_weights.items():
        if float(weight) <= 0.0:
            continue
        holdings.append(
            {
                "symbol": symbol,
                "weight": float(weight),
                "percent": float(weight) * 100.0,
            }
        )
    return holdings[:10]


def _build_recent_trades(result: BacktestResult) -> list[dict[str, Any]]:
    if result.trades.empty:
        return []

    trades = result.trades.tail(8).reset_index()
    date_column = trades.columns[0]
    return [
        {
            "date": row[date_column].strftime("%Y-%m-%d") if hasattr(row[date_column], "strftime") else str(row[date_column]),
            "turnover": float(row["turnover"]),
        }
        for _, row in trades.iterrows()
    ]


def build_backtest_snapshot(
    cfg: DictConfig | list[str] | None = None,
) -> BacktestSnapshot:
    """Run the backtest and return a rich snapshot."""
    if not isinstance(cfg, DictConfig):
        cfg = load_config(cfg)

    # Convert to a plain Python dict for uniform access
    config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    data_cfg = config.get("data", {})
    universe_cfg = config.get("universe", {})
    provider = str(data_cfg.get("provider", ""))
    symbols = universe_cfg.get("symbols", [])
    strategy_cfg = config.get("strategy", {})

    if provider != "yahoo":
        raise ValueError(f"Unsupported data provider: {provider}")
    if not symbols:
        raise ValueError("universe.symbols must contain at least one symbol")

    execution_cfg = config.get("execution", {})
    execution_model = IndiaEquitiesExecutionModel(
        brokerage_rate=float(execution_cfg["brokerage_cost"]),
        slippage_rate=float(execution_cfg["slippage"]),
    )

    strategy = build_strategy(strategy_cfg)

    backtest_cfg = config.get("backtest", {})
    rebalance_every = int(backtest_cfg.get("rebalance_every", 21))

    engine = BacktestEngine(
        price_repo=YahooPriceRepository(),
        execution_model=execution_model,
        strategy=strategy,
        rebalance_every=rebalance_every,
    )

    start_date = _parse_date(backtest_cfg["start_date"])
    if start_date is None:
        raise ValueError("backtest.start_date must be set")

    today = date.today()
    if start_date > today:
        raise ValueError(f"Start date ({start_date}) cannot be in the future. Today is {today}.")

    end_date = _parse_date(backtest_cfg.get("end_date"))
    initial_capital = float(backtest_cfg["initial_capital"])
    fetch_start = start_date - relativedelta(months=14)

    result = engine.run(
        symbols=symbols,
        start_date=fetch_start,
        end_date=end_date,
        initial_capital=initial_capital,
    )

    if result.equity_curve.empty:
        raise RuntimeError("Backtest produced no output. Check symbols/date range/data source.")

    metrics = calculate_metrics(returns=result.returns, equity_curve=result.equity_curve)
    chart_path, chart_min, chart_max = _build_chart_path(result.equity_curve)

    return BacktestSnapshot(
        config=config,
        result=result,
        metrics=metrics,
        chart_path=chart_path,
        chart_min=chart_min,
        chart_max=chart_max,
        holdings=_build_holdings(result),
        recent_trades=_build_recent_trades(result),
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        symbols=list(symbols),
    )
