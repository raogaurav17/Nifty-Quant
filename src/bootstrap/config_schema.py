from dataclasses import dataclass
from typing import List, Optional


# -------------------------
# Universe
# -------------------------


@dataclass(frozen=True)
class UniverseConfig:
    name: str
    symbols: List[str]


# -------------------------
# Data
# -------------------------


@dataclass(frozen=True)
class DataConfig:
    provider: str
    adjusted_prices: bool


# -------------------------
# Strategy
# -------------------------


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    lookback_days: int
    skip_recent_days: int
    top_k: int


# -------------------------
# Portfolio Construction
# -------------------------


@dataclass(frozen=True)
class PortfolioConfig:
    method: str
    vol_lookback_days: int
    max_weight: float
    cash_buffer: float
    target_annual_vol: float


# -------------------------
# Execution / Costs
# -------------------------


@dataclass(frozen=True)
class ExecutionConfig:
    brokerage_cost: float
    slippage: float


# -------------------------
# Backtest
# -------------------------


@dataclass(frozen=True)
class BacktestConfig:
    frequency: str
    initial_capital: float
    start_date: str
    end_date: Optional[str]


# -------------------------
# Metrics
# -------------------------


@dataclass(frozen=True)
class MetricsConfig:
    risk_free_rate: float
    annualization_factor: int


# -------------------------
# Runtime
# -------------------------


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    timezone: str
    log_level: str


# -------------------------
# Root App Config
# -------------------------


@dataclass(frozen=True)
class AppConfig:
    universe: UniverseConfig
    data: DataConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    metrics: MetricsConfig
    runtime: RuntimeConfig
