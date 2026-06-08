"""Strategy registry factory."""

from __future__ import annotations

from typing import Any, Dict

from nifty_quant.domain.strategies.base import Strategy

_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Class decorator that registers a strategy under ``name``."""
    def _decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return _decorator


def _register_defaults() -> None:
    """Populate registry with defaults."""
    from nifty_quant.domain.strategies.momentum_12_1 import Momentum12_1Strategy  # noqa: F401
    from nifty_quant.domain.strategies.arima import ARIMAStrategy  # noqa: F401

    _REGISTRY.setdefault("momentum_12_1", Momentum12_1Strategy)
    _REGISTRY.setdefault("arima", ARIMAStrategy)


def build_strategy(cfg: Dict[str, Any]) -> Strategy:
    """Instantiate a Strategy from a configuration dictionary."""
    _register_defaults()

    name = cfg.get("name")
    if not name:
        raise ValueError(
            "strategy config must contain a 'name' key. "
            f"Available strategies: {sorted(_REGISTRY)}"
        )

    cls = _REGISTRY.get(str(name))
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available strategies: {sorted(_REGISTRY)}"
        )

    # Strip 'name' from kwargs; pass everything else to the constructor
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return cls(**kwargs)


def available_strategies() -> list[str]:
    """Return the list of registered strategy names."""
    _register_defaults()
    return sorted(_REGISTRY)
