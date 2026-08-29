"""Unit tests for CachedPriceRepository."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from nifty_quant.infrastructure.data.cached_price_repository import CachedPriceRepository
from nifty_quant.interfaces.price_repository import PriceRepository


class DummyPriceRepo(PriceRepository):
    def __init__(self) -> None:
        self.call_history: list[dict] = []

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        self.call_history.append({"symbols": symbols, "start_date": start_date, "end_date": end_date})
        effective_end = end_date if end_date else date(2023, 1, 10)
        dates = pd.date_range(start_date, effective_end, freq="D")
        results = {}
        for sym in symbols:
            results[sym] = pd.DataFrame(
                {"adj_close": [100.0 + i for i in range(len(dates))], "volume": [1000] * len(dates)},
                index=dates,
            )
        return results


def test_cached_repo_cold_start_then_hit(tmp_path: Path):
    upstream = DummyPriceRepo()
    cached = CachedPriceRepository(upstream=upstream, cache_dir=tmp_path)

    # 1. Cold start query: 2023-01-01 to 2023-01-10
    res1 = cached.get_prices(["AAA.NS"], start_date=date(2023, 1, 1), end_date=date(2023, 1, 10))
    assert "AAA.NS" in res1
    assert len(res1["AAA.NS"]) == 10
    assert len(upstream.call_history) == 1

    # 2. Repeated exact query -> upstream should NOT be called again
    res2 = cached.get_prices(["AAA.NS"], start_date=date(2023, 1, 1), end_date=date(2023, 1, 10))
    assert len(res2["AAA.NS"]) == 10
    assert len(upstream.call_history) == 1

    # 3. Subset query (2023-01-03 to 2023-01-07) -> upstream should NOT be called
    res3 = cached.get_prices(["AAA.NS"], start_date=date(2023, 1, 3), end_date=date(2023, 1, 7))
    assert len(res3["AAA.NS"]) == 5
    assert len(upstream.call_history) == 1


def test_cached_repo_partial_gap_download(tmp_path: Path):
    upstream = DummyPriceRepo()
    cached = CachedPriceRepository(upstream=upstream, cache_dir=tmp_path)

    # Initial query: 2023-01-05 to 2023-01-10
    cached.get_prices(["AAA.NS"], start_date=date(2023, 1, 5), end_date=date(2023, 1, 10))
    assert len(upstream.call_history) == 1
    assert upstream.call_history[0]["start_date"] == date(2023, 1, 5)
    assert upstream.call_history[0]["end_date"] == date(2023, 1, 10)

    # Expanded query: 2023-01-01 to 2023-01-15
    # Missing gaps are [2023-01-01 to 2023-01-04] and [2023-01-11 to 2023-01-15]
    res = cached.get_prices(["AAA.NS"], start_date=date(2023, 1, 1), end_date=date(2023, 1, 15))
    assert len(res["AAA.NS"]) == 15
    # Upstream should have been called 2 more times (once for each gap)
    assert len(upstream.call_history) == 3
    called_ranges = [(c["start_date"], c["end_date"]) for c in upstream.call_history[1:]]
    assert (date(2023, 1, 1), date(2023, 1, 4)) in called_ranges
    assert (date(2023, 1, 11), date(2023, 1, 15)) in called_ranges


def test_cached_repo_force_refresh(tmp_path: Path):
    upstream = DummyPriceRepo()
    cached = CachedPriceRepository(upstream=upstream, cache_dir=tmp_path, force_refresh=True)

    # 1st call
    cached.get_prices(["AAA.NS"], start_date=date(2023, 1, 1), end_date=date(2023, 1, 5))
    assert len(upstream.call_history) == 1

    # 2nd call with force_refresh=True -> upstream MUST be called again
    cached.get_prices(["AAA.NS"], start_date=date(2023, 1, 1), end_date=date(2023, 1, 5))
    assert len(upstream.call_history) == 2


def test_cached_repo_batch_symbol_gaps(tmp_path: Path):
    upstream = DummyPriceRepo()
    cached = CachedPriceRepository(upstream=upstream, cache_dir=tmp_path)

    # Query multiple symbols with same range: should batch into 1 upstream call
    res = cached.get_prices(["AAA.NS", "BBB.NS"], start_date=date(2023, 1, 1), end_date=date(2023, 1, 5))
    assert set(res.keys()) == {"AAA.NS", "BBB.NS"}
    assert len(upstream.call_history) == 1
    assert set(upstream.call_history[0]["symbols"]) == {"AAA.NS", "BBB.NS"}
