"""Strategy registry factory and class decorator."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from nifty_quant.domain.strategies.base import Strategy

StrategyClass = TypeVar("StrategyClass", bound=type[Strategy])

_REGISTRY: dict[str, type[Strategy]] = {}


def register(name: str) -> Callable[[StrategyClass], StrategyClass]:
    """Class decorator that registers a Strategy subclass under ``name``.

    Args:
        name: Unique string identifier for the strategy.

    Returns:
        Decorator function returning the registered strategy class.
    """
    def _decorator(cls: StrategyClass) -> StrategyClass:
        _REGISTRY[name] = cls
        return cls

    return _decorator


def _register_defaults() -> None:
    """Ensure all default strategies are imported and registered."""
    import nifty_quant.domain.strategies.arima  # noqa: F401
    import nifty_quant.domain.strategies.low_vol  # noqa: F401
    import nifty_quant.domain.strategies.momentum_12_1  # noqa: F401


def get_strategy_class(name: str) -> type[Strategy]:
    """Retrieve a registered strategy class by name."""
    _register_defaults()
    cls = _REGISTRY.get(str(name))
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available strategies: {sorted(_REGISTRY)}"
        )
    return cls


def build_strategy(cfg: dict[str, Any]) -> Strategy:
    """Instantiate a Strategy from a configuration dictionary.

    Args:
        cfg: Configuration dictionary with a 'name' key and constructor kwargs.

    Returns:
        Instantiated Strategy instance.
    """
    name = cfg.get("name")
    if not name:
        _register_defaults()
        raise ValueError(
            "Strategy config must contain a 'name' key. "
            f"Available strategies: {sorted(_REGISTRY)}"
        )

    cls = get_strategy_class(str(name))
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return cls(**kwargs)


def available_strategies() -> list[str]:
    """Return the list of all registered strategy names."""
    _register_defaults()
    return sorted(_REGISTRY.keys())

