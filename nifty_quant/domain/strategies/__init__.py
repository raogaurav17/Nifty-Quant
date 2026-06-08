"""Strategy plugin package."""

from nifty_quant.domain.strategies.base import Strategy
from nifty_quant.domain.strategies.registry import build_strategy

__all__ = ["Strategy", "build_strategy"]
