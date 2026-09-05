"""Configuration schema dataclasses for Hydra and application configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UniverseConfig:
    """Configuration for universe constituent symbols."""

    name: str
    symbols: list[str] = field(default_factory=list)
    dynamic: bool = False
    timeline_file: str | None = None


@dataclass(frozen=True)
class DataConfig:
    """Configuration for price data provider."""

    provider: str
    adjusted_prices: bool
    use_cache: bool = True
    cache_dir: str = "data/cache"
    force_refresh: bool = False


@dataclass
class StrategyConfig:
    """Configuration for strategy selection and dynamic parameters."""

    name: str
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __init__(self, name: str, **kwargs: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_extra", kwargs)

    def __getattr__(self, item: str) -> Any:
        try:
            return self._extra[item]
        except KeyError:
            raise AttributeError(item) from None


@dataclass(frozen=True)
class PortfolioConfig:
    """Configuration for portfolio weighting and risk targeting."""

    method: str
    vol_lookback_days: int
    max_weight: float
    cash_buffer: float
    target_annual_vol: float


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for transaction costs and slippage."""

    brokerage_cost: float
    slippage: float


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for backtest time period, capital, and rebalance frequency."""

    frequency: str
    initial_capital: float
    start_date: str
    end_date: str | None
    rebalance_every: int = 21   # trading days between rebalances


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for performance metrics calculation."""

    risk_free_rate: float
    annualization_factor: int


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for runtime environment, seeds, and logging."""

    seed: int
    timezone: str
    log_level: str


@dataclass
class AppConfig:
    """Root application configuration tree."""

    universe: UniverseConfig
    data: DataConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    metrics: MetricsConfig
    runtime: RuntimeConfig

    def __init__(
        self,
        universe: Any,
        data: Any,
        strategy: Any,
        portfolio: Any,
        execution: Any,
        backtest: Any,
        metrics: Any,
        runtime: Any,
    ) -> None:
        """Accepts dicts or typed configuration objects."""
        self.universe = UniverseConfig(**universe) if isinstance(universe, dict) else universe
        self.data = DataConfig(**data) if isinstance(data, dict) else data
        self.strategy = StrategyConfig(**strategy) if isinstance(strategy, dict) else strategy
        self.portfolio = PortfolioConfig(**portfolio) if isinstance(portfolio, dict) else portfolio
        self.execution = ExecutionConfig(**execution) if isinstance(execution, dict) else execution
        self.backtest = BacktestConfig(**backtest) if isinstance(backtest, dict) else backtest
        self.metrics = MetricsConfig(**metrics) if isinstance(metrics, dict) else metrics
        self.runtime = RuntimeConfig(**runtime) if isinstance(runtime, dict) else runtime

