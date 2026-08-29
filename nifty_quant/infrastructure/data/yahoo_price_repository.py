"""Price repository implementation fetching market data from Yahoo Finance."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import logging
import time
from typing import Iterator

import pandas as pd
import yfinance as yf

from nifty_quant.interfaces.price_repository import PriceRepository

logger = logging.getLogger(__name__)


class YahooPriceRepository(PriceRepository):
    """Price repository backed by Yahoo Finance."""

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical price data from Yahoo Finance.

        Args:
            symbols: List of ticker symbols (e.g. ['RELIANCE.NS', 'TCS.NS']).
            start_date: Start date for historical data.
            end_date: Optional end date for historical data.

        Returns:
            Dictionary mapping symbol string to DataFrame with 'adj_close' and 'volume'.
        """
        if not symbols:
            return {}

        raw = self._download_with_retry(symbols=symbols, start_date=start_date, end_date=end_date)
        symbol_to_df = self._extract_symbol_data(raw=raw, symbols=symbols)

        # Retry missing symbols individually
        missing_symbols = [symbol for symbol in symbols if symbol not in symbol_to_df]
        for symbol in missing_symbols:
            single_raw = self._download_with_retry(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
            )
            single_data = self._extract_symbol_data(raw=single_raw, symbols=[symbol])
            if symbol in single_data:
                symbol_to_df[symbol] = single_data[symbol]
            else:
                logger.warning("No Yahoo data returned for symbol %s", symbol)

        return symbol_to_df

    def _download_with_retry(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date | None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> pd.DataFrame:
        """Download price data with retry logic and error suppression."""
        for attempt in range(1, max_attempts + 1):
            try:
                with self._suppress_yfinance_errors():
                    return yf.download(
                        tickers=symbols,
                        start=start_date.isoformat(),
                        end=end_date.isoformat() if end_date else None,
                        auto_adjust=False,
                        progress=False,
                        group_by="ticker",
                        threads=False,
                        timeout=10,
                    )
            except Exception as exc:  # pragma: no cover - depends on network behavior
                if attempt == max_attempts:
                    logger.warning(
                        "Yahoo download failed after %s attempts for %s: %s",
                        max_attempts,
                        symbols,
                        exc,
                    )
                    return pd.DataFrame()
                time.sleep(retry_delay_seconds)

        return pd.DataFrame()

    @contextmanager
    def _suppress_yfinance_errors(self) -> Iterator[None]:
        """Context manager to suppress verbose internal Yahoo Finance logging."""
        yfinance_logger = logging.getLogger("yfinance")
        previous_level = yfinance_logger.level
        yfinance_logger.setLevel(logging.CRITICAL)
        try:
            yield
        finally:
            yfinance_logger.setLevel(previous_level)

    def _extract_symbol_data(
        self,
        raw: pd.DataFrame,
        symbols: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Parse raw multi-index or single-level Yahoo DataFrame into per-symbol DataFrames."""
        if raw.empty:
            return {}

        symbol_to_df: dict[str, pd.DataFrame] = {}

        if isinstance(raw.columns, pd.MultiIndex):
            for symbol in symbols:
                if symbol not in raw.columns.get_level_values(0):
                    continue

                sym_df = raw[symbol].copy()
                if sym_df.empty:
                    continue

                adj_close_col = "Adj Close" if "Adj Close" in sym_df.columns else "Close"
                out = pd.DataFrame(
                    {
                        "adj_close": sym_df[adj_close_col],
                        "volume": sym_df.get("Volume", pd.Series(0.0, index=sym_df.index)),
                    },
                    index=sym_df.index,
                ).dropna(subset=["adj_close"])
                if not out.empty:
                    symbol_to_df[symbol] = out
        else:
            # Handle single-level columns for single symbol request
            adj_close_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            symbol = symbols[0]
            out = pd.DataFrame(
                {
                    "adj_close": raw[adj_close_col],
                    "volume": raw.get("Volume", pd.Series(0.0, index=raw.index)),
                },
                index=raw.index,
            ).dropna(subset=["adj_close"])
            if not out.empty:
                symbol_to_df[symbol] = out

        return symbol_to_df

