"""Factory function for creating UniverseProvider instances."""

from __future__ import annotations

from typing import Any

from nifty_quant.bootstrap.config_schema import UniverseConfig
from nifty_quant.domain.universe.dynamic_universe import DynamicUniverseProvider
from nifty_quant.interfaces.universe_provider import UniverseProvider


def build_universe(universe_cfg: dict[str, Any] | UniverseConfig) -> UniverseProvider:
    """Build a DynamicUniverseProvider from configuration.

    All universe configurations are dynamic and survivorship bias-free.
    The timeline_file key should point to the historical reconstitution JSON.
    """
    if isinstance(universe_cfg, UniverseConfig):
        cfg_dict = {
            "name": universe_cfg.name,
            "timeline_file": universe_cfg.timeline_file,
        }
    else:
        cfg_dict = dict(universe_cfg)

    timeline_file = cfg_dict.get("timeline_file") or "conf/universe/nifty50_timeline.json"
    return DynamicUniverseProvider(
        timeline_source=timeline_file,
        name=str(cfg_dict.get("name", "nifty50")),
    )
