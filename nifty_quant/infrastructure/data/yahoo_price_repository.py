from datetime import date
from typing import Dict, List

import pandas as pd
import yfinance as yf

from nifty_quant.interfaces.price_repository import PriceRepository


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

        raw = yf.download(
            tickers=symbols,
            start=start_date.isoformat(),
            end=end_date.isoformat() if end_date else None,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )

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
