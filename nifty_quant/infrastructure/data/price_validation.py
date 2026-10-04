"""Validation helpers for historical price data."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_REVERSAL_RATIO = 3.0


def remove_isolated_price_spikes(
    prices: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Remove one-day price spikes that immediately revert.

    A genuine split or bonus should be reflected consistently in an adjusted
    price series. An isolated spike followed by a return to the prior level is
    therefore treated as a bad source-data row. Zero-volume rows are included
    when volume data is available; an all-zero volume series means volume is
    unavailable and is not used as a filter.
    """
    if prices.empty or "adj_close" not in prices.columns:
        return prices

    clean = prices.sort_index().copy()
    close = pd.to_numeric(clean["adj_close"], errors="coerce")
    previous = close.shift(1)
    following = close.shift(-1)
    valid_prices = close.gt(0) & previous.gt(0) & following.gt(0)

    spike_ratio = close / previous
    reversal_ratio = following / close
    reverse_from_high = (
        spike_ratio.ge(_REVERSAL_RATIO)
        & reversal_ratio.le(1.0 / _REVERSAL_RATIO)
    )
    reverse_from_low = (
        spike_ratio.le(1.0 / _REVERSAL_RATIO)
        & reversal_ratio.ge(_REVERSAL_RATIO)
    )
    suspicious = valid_prices & (reverse_from_high | reverse_from_low)

    if "volume" in clean.columns:
        volume = pd.to_numeric(clean["volume"], errors="coerce")
        volume_available = volume.fillna(0).gt(0).any()
        if volume_available:
            suspicious &= volume.le(0).fillna(True)

    invalid_dates = clean.index[suspicious]
    if len(invalid_dates):
        logger.warning(
            "Ignoring %d isolated price spike(s) for %s: %s",
            len(invalid_dates),
            symbol,
            ", ".join(str(timestamp.date()) for timestamp in invalid_dates),
        )
        clean = clean.drop(index=invalid_dates)

    return clean
