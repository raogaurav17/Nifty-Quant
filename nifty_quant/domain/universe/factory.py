"""Factory function for creating UniverseProvider instances."""

from __future__ import annotations

from typing import Any

from nifty_quant.bootstrap.config_schema import UniverseConfig
from nifty_quant.domain.universe.dynamic_universe import DynamicUniverseProvider
from nifty_quant.domain.universe.static_universe import StaticUniverseProvider
from nifty_quant.interfaces.universe_provider import UniverseProvider


def build_universe(universe_cfg: dict[str, Any] | UniverseConfig) -> UniverseProvider:
    """Build a StaticUniverseProvider or DynamicUniverseProvider from configuration."""
    if isinstance(universe_cfg, UniverseConfig):
        cfg_dict = {
            "name": universe_cfg.name,
            "symbols": universe_cfg.symbols,
            "dynamic": universe_cfg.dynamic,
            "timeline_file": universe_cfg.timeline_file,
        }
    else:
        cfg_dict = dict(universe_cfg)

    is_dynamic = bool(cfg_dict.get("dynamic", False))
    timeline_file = cfg_dict.get("timeline_file")

    if is_dynamic or timeline_file:
        timeline_path = timeline_file or "conf/universe/nifty50_timeline.json"
        return DynamicUniverseProvider(
            timeline_source=timeline_path,
            name=str(cfg_dict.get("name", "nifty50_dynamic")),
        )

    symbols = cfg_dict.get("symbols", [])
    return StaticUniverseProvider(
        symbols=list(symbols),
        name=str(cfg_dict.get("name", "static")),
    )
