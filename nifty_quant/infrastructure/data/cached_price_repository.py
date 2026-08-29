"""Cached price repository wrapping upstream data provider with interval tracking."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import logging
from pathlib import Path

import pandas as pd

from nifty_quant.infrastructure.data.date_intervals import DateInterval, IntervalRegistry
from nifty_quant.infrastructure.data.local_price_store import LocalPriceStore
from nifty_quant.interfaces.price_repository import PriceRepository

logger = logging.getLogger(__name__)


class CachedPriceRepository(PriceRepository):
    """Price repository with local persistence, interval math, and partial gap fetching."""

    def __init__(
        self,
        upstream: PriceRepository,
        cache_dir: Path | str = "data/cache",
        force_refresh: bool = False,
    ) -> None:
        self.upstream = upstream
        self.cache_dir = Path(cache_dir)
        self.force_refresh = force_refresh

        self.registry = IntervalRegistry(self.cache_dir / "intervals.json")
        self.store = LocalPriceStore(self.cache_dir)

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical prices for symbols, downloading only missing date intervals.

        Args:
            symbols: List of ticker symbols to query.
            start_date: Starting date for price history.
            end_date: Optional ending date for price history (defaults to today).

        Returns:
            Dictionary mapping symbol string to DataFrame containing 'adj_close' and 'volume'.
        """
        if not symbols:
            return {}

        effective_end = end_date if end_date is not None else date.today()
        target_interval = DateInterval(start_date, effective_end)

        # 1. Identify missing gaps for each symbol
        # Group symbols by gap interval to optimize batch fetching
        gap_to_symbols: dict[DateInterval, list[str]] = defaultdict(list)

        for symbol in symbols:
            if self.force_refresh:
                missing_gaps = [target_interval]
            else:
                missing_gaps = self.registry.get_missing_gaps(symbol, target_interval)

            for gap in missing_gaps:
                gap_to_symbols[gap].append(symbol)

        # 2. Fetch missing gaps from upstream in batches
        registry_modified = False
        for gap, gap_symbols in gap_to_symbols.items():
            logger.info(
                "Cache miss for %d symbol(s) in range [%s to %s]. Fetching from upstream...",
                len(gap_symbols),
                gap.start,
                gap.end,
            )
            try:
                fetched_data = self.upstream.get_prices(
                    symbols=gap_symbols,
                    start_date=gap.start,
                    end_date=gap.end,
                )
            except Exception as exc:
                logger.error(
                    "Upstream fetch failed for symbols %s in range [%s, %s]: %s",
                    gap_symbols,
                    gap.start,
                    gap.end,
                    exc,
                )
                fetched_data = {}

            for symbol in gap_symbols:
                if symbol in fetched_data and not fetched_data[symbol].empty:
                    self.store.upsert(symbol, fetched_data[symbol])

                # Mark interval as registered so we do not redundantly re-query
                self.registry.add_interval(symbol, gap)
                registry_modified = True

        if registry_modified:
            self.registry.save()

        # 3. Read complete requested range [start_date, effective_end] from local store
        results: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.store.read(symbol, start_date=start_date, end_date=effective_end)
            if not df.empty:
                results[symbol] = df
            else:
                logger.debug("No local data found for symbol %s in requested range.", symbol)

        return results
