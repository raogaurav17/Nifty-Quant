"""Local price storage manager for per-symbol historical market data."""

from __future__ import annotations

from datetime import date
import logging
from pathlib import Path
import tempfile
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


class LocalPriceStore:
    """Manages persistent, consolidated time-series files per symbol on disk."""

    def __init__(
        self,
        base_dir: Path | str,
        storage_format: Literal["auto", "parquet", "csv"] = "auto",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.prices_dir = self.base_dir / "prices"
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        self._storage_format = self._detect_format(storage_format)

    def _detect_format(self, requested: str) -> str:
        if requested in ("parquet", "csv"):
            return requested

        # Auto-detect: check if parquet writing is supported
        try:
            test_df = pd.DataFrame({"a": [1]})
            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
                test_df.to_parquet(tmp.name)
            return "parquet"
        except Exception:
            return "csv"

    def _get_file_path(self, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        extension = "parquet" if self._storage_format == "parquet" else "csv"
        return self.prices_dir / f"{safe_symbol}.{extension}"

    def read(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Read historical price data for symbol filtered to [start_date, end_date]."""
        file_path = self._get_file_path(symbol)
        if not file_path.exists():
            # Also check if opposite format exists (e.g. legacy csv or parquet)
            alt_ext = "csv" if file_path.suffix == ".parquet" else "parquet"
            alt_path = file_path.with_suffix(f".{alt_ext}")
            if alt_path.exists():
                file_path = alt_path
            else:
                return pd.DataFrame()

        try:
            if file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        except Exception as exc:
            logger.warning("Failed to read cached price file %s: %s", file_path, exc)
            return pd.DataFrame()

        if df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone for clean date comparison
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()

        if start_date is not None:
            df = df[df.index.date >= start_date]
        if end_date is not None:
            df = df[df.index.date <= end_date]

        return df

    def upsert(self, symbol: str, new_df: pd.DataFrame) -> None:
        """Merge new price data into the existing symbol file, deduplicating by date."""
        if new_df.empty:
            return

        clean_new = new_df.copy()
        if not isinstance(clean_new.index, pd.DatetimeIndex):
            clean_new.index = pd.to_datetime(clean_new.index)
        if clean_new.index.tz is not None:
            clean_new.index = clean_new.index.tz_localize(None)

        existing_df = self.read(symbol)
        if not existing_df.empty:
            combined = pd.concat([existing_df, clean_new])
        else:
            combined = clean_new

        # Deduplicate on index (keeping newest) and sort
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        file_path = self._get_file_path(symbol)
        temp_file = file_path.with_suffix(f".tmp{file_path.suffix}")

        try:
            if file_path.suffix == ".parquet":
                combined.to_parquet(temp_file)
            else:
                combined.to_csv(temp_file)
            temp_file.replace(file_path)
        except Exception as exc:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed writing price data for {symbol} to {file_path}: {exc}") from exc
