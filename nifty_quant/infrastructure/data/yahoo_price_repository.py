from datetime import date
from contextlib import contextmanager
import logging
import time
from typing import Dict, List

import pandas as pd
import yfinance as yf

from nifty_quant.interfaces.price_repository import PriceRepository


logger = logging.getLogger(__name__)


class YahooPriceRepository(PriceRepository):
    """
    Price repository backed by Yahoo Finance.
    """

    def get_prices(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date | None = None,
    ) -> Dict[str, pd.DataFrame]:
        if not symbols:
            return {}

        raw = self._download_with_retry(symbols=symbols, start_date=start_date, end_date=end_date)
        symbol_to_df = self._extract_symbol_data(raw=raw, symbols=symbols)

        # Batch requests can intermittently miss individual symbols; retry them one-by-one.
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
        symbols: List[str],
        start_date: date,
        end_date: date | None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> pd.DataFrame:
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
    def _suppress_yfinance_errors(self):
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
        symbols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        if raw.empty:
            return {}

        symbol_to_df: Dict[str, pd.DataFrame] = {}

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
            # yfinance returns single-level columns when only one symbol is requested.
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
