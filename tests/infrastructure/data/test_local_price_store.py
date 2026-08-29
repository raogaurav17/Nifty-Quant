"""Unit tests for LocalPriceStore."""

from datetime import date
from pathlib import Path
import pandas as pd

from nifty_quant.infrastructure.data.local_price_store import LocalPriceStore


def test_local_price_store_upsert_and_read(tmp_path: Path):
    store = LocalPriceStore(tmp_path, storage_format="csv")

    dates1 = pd.date_range("2023-01-01", periods=3, freq="D")
    df1 = pd.DataFrame({"adj_close": [100.0, 101.0, 102.0], "volume": [10, 20, 30]}, index=dates1)

    store.upsert("TEST.NS", df1)

    read_all = store.read("TEST.NS")
    assert len(read_all) == 3
    assert float(read_all.iloc[0]["adj_close"]) == 100.0

    # Read with date filtering
    filtered = store.read("TEST.NS", start_date=date(2023, 1, 2), end_date=date(2023, 1, 2))
    assert len(filtered) == 1
    assert float(filtered.iloc[0]["adj_close"]) == 101.0


def test_local_price_store_deduplication_and_sorting(tmp_path: Path):
    store = LocalPriceStore(tmp_path, storage_format="csv")

    dates1 = pd.date_range("2023-01-01", periods=3, freq="D")
    df1 = pd.DataFrame({"adj_close": [100.0, 101.0, 102.0], "volume": [10, 20, 30]}, index=dates1)
    store.upsert("TEST.NS", df1)

    # Overlapping write with updated value for 2023-01-03 and new date 2023-01-04
    dates2 = pd.date_range("2023-01-03", periods=2, freq="D")
    df2 = pd.DataFrame({"adj_close": [105.0, 106.0], "volume": [50, 60]}, index=dates2)
    store.upsert("TEST.NS", df2)

    combined = store.read("TEST.NS")
    assert len(combined) == 4
    # The updated value for 2023-01-03 should be kept (105.0)
    assert float(combined.loc["2023-01-03"]["adj_close"]) == 105.0
    assert float(combined.loc["2023-01-04"]["adj_close"]) == 106.0


def test_local_price_store_nonexistent_symbol(tmp_path: Path):
    store = LocalPriceStore(tmp_path)
    df = store.read("NONEXISTENT.NS")
    assert df.empty
