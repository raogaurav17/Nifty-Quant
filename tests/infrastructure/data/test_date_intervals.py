"""Unit tests for date interval math and registry."""

from datetime import date
from pathlib import Path
import pytest

from nifty_quant.infrastructure.data.date_intervals import (
    DateInterval,
    IntervalRegistry,
    merge_intervals,
    subtract_intervals,
)


def test_date_interval_validation():
    iv = DateInterval(date(2023, 1, 1), date(2023, 1, 10))
    assert iv.start == date(2023, 1, 1)
    assert iv.end == date(2023, 1, 10)

    with pytest.raises(ValueError, match="must be <= end"):
        DateInterval(date(2023, 1, 10), date(2023, 1, 1))


def test_date_interval_methods():
    iv1 = DateInterval(date(2023, 1, 1), date(2023, 1, 10))
    iv2 = DateInterval(date(2023, 1, 11), date(2023, 1, 20))  # adjacent
    iv3 = DateInterval(date(2023, 1, 5), date(2023, 1, 15))   # overlapping
    iv4 = DateInterval(date(2023, 1, 15), date(2023, 1, 25))  # disjoint with iv1

    assert iv1.contains_date(date(2023, 1, 5))
    assert not iv1.contains_date(date(2023, 1, 15))

    assert iv1.overlaps_or_adjacent(iv2)
    assert iv1.overlaps_or_adjacent(iv3)
    assert not iv1.overlaps_or_adjacent(iv4)

    merged = iv1.union(iv2)
    assert merged == DateInterval(date(2023, 1, 1), date(2023, 1, 20))

    inter = iv1.intersection(iv3)
    assert inter == DateInterval(date(2023, 1, 5), date(2023, 1, 10))
    assert iv1.intersection(iv4) is None

    # Serialization
    d = iv1.to_dict()
    assert d == {"start": "2023-01-01", "end": "2023-01-10"}
    assert DateInterval.from_dict(d) == iv1


def test_merge_intervals():
    assert merge_intervals([]) == []

    intervals = [
        DateInterval(date(2023, 1, 15), date(2023, 1, 20)),
        DateInterval(date(2023, 1, 1), date(2023, 1, 10)),
        DateInterval(date(2023, 1, 9), date(2023, 1, 16)),
    ]
    merged = merge_intervals(intervals)
    assert len(merged) == 1
    assert merged[0] == DateInterval(date(2023, 1, 1), date(2023, 1, 20))

    # Disjoint intervals
    disjoint = [
        DateInterval(date(2023, 1, 1), date(2023, 1, 10)),
        DateInterval(date(2023, 2, 1), date(2023, 2, 10)),
    ]
    assert merge_intervals(disjoint) == disjoint


def test_subtract_intervals_empty_covered():
    target = DateInterval(date(2023, 1, 1), date(2023, 1, 31))
    gaps = subtract_intervals(target, [])
    assert gaps == [target]


def test_subtract_intervals_fully_covered():
    target = DateInterval(date(2023, 1, 10), date(2023, 1, 20))
    covered = [DateInterval(date(2023, 1, 1), date(2023, 1, 31))]
    gaps = subtract_intervals(target, covered)
    assert gaps == []


def test_subtract_intervals_middle_covered():
    target = DateInterval(date(2023, 1, 1), date(2023, 1, 31))
    covered = [DateInterval(date(2023, 1, 10), date(2023, 1, 20))]
    gaps = subtract_intervals(target, covered)
    assert gaps == [
        DateInterval(date(2023, 1, 1), date(2023, 1, 9)),
        DateInterval(date(2023, 1, 21), date(2023, 1, 31)),
    ]


def test_subtract_intervals_multiple_covered_segments():
    target = DateInterval(date(2023, 1, 1), date(2023, 5, 31))
    covered = [
        DateInterval(date(2023, 2, 1), date(2023, 2, 28)),
        DateInterval(date(2023, 4, 1), date(2023, 4, 30)),
    ]
    gaps = subtract_intervals(target, covered)
    assert gaps == [
        DateInterval(date(2023, 1, 1), date(2023, 1, 31)),
        DateInterval(date(2023, 3, 1), date(2023, 3, 31)),
        DateInterval(date(2023, 5, 1), date(2023, 5, 31)),
    ]


def test_interval_registry_persistence(tmp_path: Path):
    reg_file = tmp_path / "intervals.json"
    reg = IntervalRegistry(reg_file)

    reg.add_interval("RELIANCE.NS", DateInterval(date(2023, 1, 1), date(2023, 1, 15)))
    reg.add_interval("RELIANCE.NS", DateInterval(date(2023, 1, 16), date(2023, 1, 31)))
    reg.save()

    # Re-load from disk
    reg2 = IntervalRegistry(reg_file)
    intervals = reg2.get_intervals("RELIANCE.NS")
    assert len(intervals) == 1
    assert intervals[0] == DateInterval(date(2023, 1, 1), date(2023, 1, 31))

    # Check missing gaps
    target = DateInterval(date(2023, 1, 1), date(2023, 2, 15))
    gaps = reg2.get_missing_gaps("RELIANCE.NS", target)
    assert gaps == [DateInterval(date(2023, 2, 1), date(2023, 2, 15))]
