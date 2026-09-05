from datetime import date
from pathlib import Path
import pytest

from nifty_quant.domain.universe.dynamic_universe import DynamicUniverseProvider
from nifty_quant.domain.universe.factory import build_universe
from nifty_quant.domain.universe.static_universe import StaticUniverseProvider


TIMELINE_PATH = Path(__file__).resolve().parents[2] / "conf" / "universe" / "nifty50_timeline.json"


@pytest.fixture
def dynamic_universe():
    return DynamicUniverseProvider(timeline_source=TIMELINE_PATH)


def test_baseline_constituents_count(dynamic_universe):
    constituents = dynamic_universe.get_constituents(date(2026, 3, 15))
    assert len(constituents) == 50
    assert "TRENT.NS" in constituents
    assert "BEL.NS" in constituents
    assert "JIOFIN.NS" in constituents


def test_constituents_pre_and_post_sept_2024(dynamic_universe):
    # After Sept 30, 2024: Trent & BEL are in; Divi's & LTIM are out
    post_rebalance = dynamic_universe.get_constituents(date(2024, 10, 15))
    assert len(post_rebalance) == 50
    assert "TRENT.NS" in post_rebalance
    assert "BEL.NS" in post_rebalance
    assert "DIVISLAB.NS" not in post_rebalance
    assert "LTIM.NS" not in post_rebalance

    # Before Sept 30, 2024: Divi's & LTIM were in; Trent & BEL were NOT
    pre_rebalance = dynamic_universe.get_constituents(date(2024, 8, 15))
    assert len(pre_rebalance) == 50
    assert "DIVISLAB.NS" in pre_rebalance
    assert "LTIM.NS" in pre_rebalance
    assert "TRENT.NS" not in pre_rebalance
    assert "BEL.NS" not in pre_rebalance


def test_hdfc_merger_reconstitution_july_2023(dynamic_universe):
    # Prior to July 13, 2023: HDFC was in; LTIM was not
    pre_merger = dynamic_universe.get_constituents(date(2023, 6, 1))
    assert len(pre_merger) == 50
    assert "HDFC.NS" in pre_merger
    assert "LTIM.NS" not in pre_merger

    # After July 13, 2023: LTIM entered; HDFC exited
    post_merger = dynamic_universe.get_constituents(date(2023, 8, 1))
    assert len(post_merger) == 50
    assert "LTIM.NS" in post_merger
    assert "HDFC.NS" not in post_merger


def test_all_historical_reconstitutions_maintain_exact_50_stocks(dynamic_universe):
    # Test every single event effective date and days before/after
    events = dynamic_universe.get_events()
    assert len(events) >= 10

    for event in events:
        eff_dt = date.fromisoformat(event["effective_date"])
        # Exactly on effective date
        assert len(dynamic_universe.get_constituents(eff_dt)) == 50
        # Check added symbols are present
        for added_sym in event["added"]:
            assert added_sym in dynamic_universe.get_constituents(eff_dt)
        # Check removed symbols are NOT present
        for rem_sym in event["removed"]:
            assert rem_sym not in dynamic_universe.get_constituents(eff_dt)


def test_get_all_symbols_union(dynamic_universe):
    all_symbols = dynamic_universe.get_all_symbols(date(2020, 1, 1), date(2026, 3, 15))
    # Should include current 50 + past constituents (e.g. HDFC, DIVISLAB, YESBANK, IOC, GAIL, etc.)
    assert len(all_symbols) > 50
    assert "HDFC.NS" in all_symbols
    assert "IOC.NS" in all_symbols
    assert "GAIL.NS" in all_symbols
    assert "YESBANK.NS" in all_symbols
    assert "TRENT.NS" in all_symbols


def test_factory_build_universe():
    static_uni = build_universe({"name": "nifty50", "symbols": ["AAA.NS", "BBB.NS"]})
    assert isinstance(static_uni, StaticUniverseProvider)
    assert not static_uni.is_dynamic
    assert static_uni.get_constituents(date.today()) == ["AAA.NS", "BBB.NS"]

    dynamic_uni = build_universe({
        "name": "nifty50_dynamic",
        "dynamic": True,
        "timeline_file": str(TIMELINE_PATH),
    })
    assert isinstance(dynamic_uni, DynamicUniverseProvider)
    assert dynamic_uni.is_dynamic
