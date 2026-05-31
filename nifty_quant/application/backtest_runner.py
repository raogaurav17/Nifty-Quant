from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nifty_quant.bootstrap.config_schema import AppConfig
from nifty_quant.domain.backtest.engine import BacktestEngine
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
    return date.fromisoformat(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(loaded).__name__}")
    return loaded


def _parse_override_value(value: str) -> Any:
    normalized = value.strip()
    if normalized.startswith(("'", '"')) and normalized.endswith(("'", '"')) and len(normalized) >= 2:
        return normalized[1:-1]

    lowered = normalized.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    numeric_value = normalized.replace("_", "")
    try:
        if any(marker in numeric_value for marker in (".", "e", "E")):
            return float(numeric_value)
        return int(numeric_value)
    except ValueError:
        return normalized


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected key=value.")

    key_path, raw_value = override.split("=", 1)
    if not key_path:
        raise ValueError(f"Invalid override '{override}'. Expected key=value.")

    target = config
    keys = key_path.split(".")
    for key in keys[:-1]:
        next_value = target.get(key)
        if next_value is None:
            next_value = {}
            target[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot apply override '{override}' because '{key}' is not a mapping")
        target = next_value

    target[keys[-1]] = _parse_override_value(raw_value)


def load_config(overrides: list[str] | None = None) -> dict[str, Any]:
    config_root = Path(__file__).resolve().parents[2] / "conf"
    root_config = _load_yaml(config_root / "config.yaml")
    defaults = root_config.get("defaults", [])

    composed_config: dict[str, Any] = {}
    for default_entry in defaults:
        if default_entry == "_self_":
            continue
        if not isinstance(default_entry, dict) or len(default_entry) != 1:
            raise ValueError(f"Unsupported default entry in config.yaml: {default_entry!r}")

        group_name, config_name = next(iter(default_entry.items()))
        composed_config[group_name] = _load_yaml(config_root / group_name / f"{config_name}.yaml")

    root_values = {key: value for key, value in root_config.items() if key != "defaults"}
    composed_config.update(root_values)

    for override in overrides or []:
        _apply_override(composed_config, override)

    return composed_config


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


def build_backtest_snapshot(overrides: list[str] | None = None) -> BacktestSnapshot:
    config = load_config(overrides)
    _ = AppConfig(**deepcopy(config))

    data_cfg = config.get("data", {})
    universe_cfg = config.get("universe", {})
    provider = str(data_cfg.get("provider", ""))
    symbols = universe_cfg.get("symbols", [])
    strategy_cfg = config.get("strategy", {})
    portfolio_cfg = config.get("portfolio", {})

    if provider != "yahoo":
        raise ValueError(f"Unsupported data provider: {provider}")
    if not symbols:
        raise ValueError("universe.symbols must contain at least one symbol")

    execution_cfg = config.get("execution", {})
    execution_model = IndiaEquitiesExecutionModel(
        brokerage_rate=float(execution_cfg["brokerage_cost"]),
        slippage_rate=float(execution_cfg["slippage"]),
    )

    engine = BacktestEngine(
        price_repo=YahooPriceRepository(),
        execution_model=execution_model,
        lookback_days=int(strategy_cfg.get("lookback_days", 252)),
        skip_recent_days=int(strategy_cfg.get("skip_recent_days", 21)),
        top_k=int(strategy_cfg.get("top_k", 10)),
        vol_lookback_days=int(portfolio_cfg.get("vol_lookback_days", 60)),
        max_weight=float(portfolio_cfg.get("max_weight", 0.10)),
        cash_buffer=float(portfolio_cfg.get("cash_buffer", 0.05)),
        target_annual_vol=float(portfolio_cfg.get("target_annual_vol", 0.10)),
    )

    backtest_cfg = config.get("backtest", {})
    start_date = _parse_date(backtest_cfg["start_date"])
    if start_date is None:
        raise ValueError("backtest.start_date must be set")

    # Validate start_date is not in the future
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
