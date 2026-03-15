from datetime import date
from unittest.mock import Mock

import pandas as pd

from nifty_quant.infrastructure.data.yahoo_price_repository import YahooPriceRepository


def test_get_prices_returns_empty_for_no_symbols():
    repo = YahooPriceRepository()

    result = repo.get_prices(symbols=[], start_date=date(2020, 1, 1), end_date=None)

    assert result == {}


def test_get_prices_parses_multi_symbol_multiindex(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=3, freq="D")

    raw = pd.DataFrame(
        {
            ("AAA.NS", "Adj Close"): [100.0, 101.0, 102.0],
            ("AAA.NS", "Volume"): [1000, 1100, 1200],
            ("BBB.NS", "Adj Close"): [200.0, 201.0, 202.0],
            ("BBB.NS", "Volume"): [2000, 2100, 2200],
        },
        index=dates,
    )

    called = {}

    def fake_download(**kwargs):
        called.update(kwargs)
        return raw

    monkeypatch.setattr(
        "nifty_quant.infrastructure.data.yahoo_price_repository.yf.download",
        fake_download,
    )

    repo = YahooPriceRepository()
    result = repo.get_prices(
        symbols=["AAA.NS", "BBB.NS"],
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 10),
    )

    assert set(result.keys()) == {"AAA.NS", "BBB.NS"}
    assert list(result["AAA.NS"].columns) == ["adj_close", "volume"]
    assert list(result["BBB.NS"].columns) == ["adj_close", "volume"]
    assert float(result["AAA.NS"].iloc[0]["adj_close"]) == 100.0
    assert float(result["BBB.NS"].iloc[-1]["adj_close"]) == 202.0
    assert int(result["AAA.NS"].iloc[1]["volume"]) == 1100
    assert called["tickers"] == ["AAA.NS", "BBB.NS"]


def test_get_prices_parses_single_symbol_shape(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Adj Close": [300.0, 301.0, 302.0],
            "Volume": [3000, 3100, 3200],
        },
        index=dates,
    )

    def fake_download(**_kwargs):
        return raw

    monkeypatch.setattr(
        "nifty_quant.infrastructure.data.yahoo_price_repository.yf.download",
        fake_download,
    )

    repo = YahooPriceRepository()
    result = repo.get_prices(
        symbols=["SINGLE.NS"],
        start_date=date(2020, 1, 1),
        end_date=None,
    )

    assert set(result.keys()) == {"SINGLE.NS"}
    assert list(result["SINGLE.NS"].columns) == ["adj_close", "volume"]
    assert float(result["SINGLE.NS"].iloc[2]["adj_close"]) == 302.0


def test_get_prices_falls_back_to_close_when_adj_close_missing(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    raw = pd.DataFrame(
        {
            "Close": [400.0, 401.0],
            "Volume": [4000, 4100],
        },
        index=dates,
    )

    def fake_download(**_kwargs):
        return raw

    monkeypatch.setattr(
        "nifty_quant.infrastructure.data.yahoo_price_repository.yf.download",
        fake_download,
    )

    repo = YahooPriceRepository()
    result = repo.get_prices(
        symbols=["CLOSE.NS"],
        start_date=date(2020, 1, 1),
        end_date=None,
    )

    assert float(result["CLOSE.NS"].iloc[0]["adj_close"]) == 400.0
    assert int(result["CLOSE.NS"].iloc[1]["volume"]) == 4100


def test_get_prices_returns_empty_when_yahoo_returns_empty_frame(monkeypatch):
    raw = pd.DataFrame()

    def fake_download(**_kwargs):
        return raw

    monkeypatch.setattr(
        "nifty_quant.infrastructure.data.yahoo_price_repository.yf.download",
        fake_download,
    )

    repo = YahooPriceRepository()
    result = repo.get_prices(
        symbols=["AAA.NS"],
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 10),
    )

    assert result == {}


def test_get_prices_defaults_volume_to_zero_when_missing(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    raw = pd.DataFrame(
        {
            "Adj Close": [500.0, 501.0],
        },
        index=dates,
    )

    def fake_download(**_kwargs):
        return raw

    monkeypatch.setattr(
        "nifty_quant.infrastructure.data.yahoo_price_repository.yf.download",
        fake_download,
    )

    repo = YahooPriceRepository()
    result = repo.get_prices(
        symbols=["NOVOL.NS"],
        start_date=date(2020, 1, 1),
        end_date=None,
    )

    assert set(result.keys()) == {"NOVOL.NS"}
    assert list(result["NOVOL.NS"]["volume"]) == [0.0, 0.0]


def test_get_prices_refetches_missing_symbol_individually(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    multi_raw = pd.DataFrame(
        {
            ("AAA.NS", "Adj Close"): [100.0, 101.0],
            ("AAA.NS", "Volume"): [1000, 1100],
        },
        index=dates,
    )
    single_bbb_raw = pd.DataFrame(
        {
            "Adj Close": [200.0, 201.0],
            "Volume": [2000, 2100],
        },
        index=dates,
    )

    calls: list[list[str]] = []

    def fake_download(**kwargs):
        tickers = kwargs["tickers"]
        calls.append(list(tickers))
        if list(tickers) == ["AAA.NS", "BBB.NS"]:
            return multi_raw
        if list(tickers) == ["BBB.NS"]:
            return single_bbb_raw
        return pd.DataFrame()

    monkeypatch.setattr(
        "nifty_quant.infrastructure.data.yahoo_price_repository.yf.download",
        fake_download,
    )

    repo = YahooPriceRepository()
    result = repo.get_prices(
        symbols=["AAA.NS", "BBB.NS"],
        start_date=date(2020, 1, 1),
        end_date=None,
    )

    assert set(result.keys()) == {"AAA.NS", "BBB.NS"}
    assert calls == [["AAA.NS", "BBB.NS"], ["BBB.NS"]]


def test_download_retries_after_exception(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    raw = pd.DataFrame(
        {
            "Adj Close": [300.0, 301.0],
            "Volume": [3000, 3100],
        },
        index=dates,
    )

    fake_download = Mock(side_effect=[ConnectionError("network"), raw])

    monkeypatch.setattr(
        "nifty_quant.infrastructure.data.yahoo_price_repository.yf.download",
        fake_download,
    )

    repo = YahooPriceRepository()
    result = repo.get_prices(
        symbols=["AAA.NS"],
        start_date=date(2020, 1, 1),
        end_date=None,
    )

    assert set(result.keys()) == {"AAA.NS"}
    assert fake_download.call_count == 2
