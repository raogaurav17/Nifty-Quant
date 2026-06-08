from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class UniverseConfig:
    name: str
    symbols: List[str]


@dataclass(frozen=True)
class DataConfig:
    provider: str
    adjusted_prices: bool


@dataclass
class StrategyConfig:
    name: str
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __init__(self, name: str, **kwargs: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_extra", kwargs)

    def __getattr__(self, item: str) -> Any:  # forward unknown attrs to _extra
        try:
            return self._extra[item]
        except KeyError:
            raise AttributeError(item) from None


@dataclass(frozen=True)
class PortfolioConfig:
    method: str
    vol_lookback_days: int
    max_weight: float
    cash_buffer: float
    target_annual_vol: float


@dataclass(frozen=True)
class ExecutionConfig:
    brokerage_cost: float
    slippage: float


@dataclass(frozen=True)
class BacktestConfig:
    frequency: str
    initial_capital: float
    start_date: str
    end_date: Optional[str]
    rebalance_every: int = 21   # trading days between rebalances


@dataclass(frozen=True)
class MetricsConfig:
    risk_free_rate: float
    annualization_factor: int


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    timezone: str
    log_level: str


@dataclass
class AppConfig:
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
        """Accepts dicts or typed objects."""
        self.universe = UniverseConfig(**universe) if isinstance(universe, dict) else universe
        self.data = DataConfig(**data) if isinstance(data, dict) else data
        self.strategy = StrategyConfig(**strategy) if isinstance(strategy, dict) else strategy
        self.portfolio = PortfolioConfig(**portfolio) if isinstance(portfolio, dict) else portfolio
        self.execution = ExecutionConfig(**execution) if isinstance(execution, dict) else execution
        self.backtest = BacktestConfig(**backtest) if isinstance(backtest, dict) else backtest
        self.metrics = MetricsConfig(**metrics) if isinstance(metrics, dict) else metrics
        self.runtime = RuntimeConfig(**runtime) if isinstance(runtime, dict) else runtime
