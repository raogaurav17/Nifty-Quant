"""Static universe provider with a fixed constituent list."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from nifty_quant.interfaces.universe_provider import UniverseProvider


class StaticUniverseProvider(UniverseProvider):
    """Provides a fixed, time-invariant list of constituent symbols."""

    def __init__(self, symbols: list[str], name: str = "static") -> None:
        if not symbols:
            raise ValueError("Static universe must contain at least one symbol.")
        self._name = name
        self._symbols = sorted(list(set(symbols)))

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_dynamic(self) -> bool:
        return False

    def get_constituents(self, as_of: date | pd.Timestamp) -> list[str]:
        return list(self._symbols)

    def get_all_symbols(
        self,
        start_date: date | pd.Timestamp,
        end_date: date | pd.Timestamp | None = None,
    ) -> list[str]:
        return list(self._symbols)

    def get_events(
        self,
        start_date: date | pd.Timestamp | None = None,
        end_date: date | pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        return []
