from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, Iterable, List

import pandas as pd


class PriceRepository(ABC):
    """
    Abstract interface for price data access.
    """

    @abstractmethod
    def get_prices(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns a dict:
        {
            "RELIANCE.NS": DataFrame(index=date, columns=[adj_close, volume]),
            ...
        }
        """
        raise NotImplementedError
