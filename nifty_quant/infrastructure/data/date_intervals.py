"""Date interval arithmetic and registry for tracking downloaded market data ranges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DateInterval:
    """Represents a closed calendar date range [start, end]."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Interval start ({self.start}) must be <= end ({self.end})")

    def contains_date(self, d: date) -> bool:
        """Check if date d is within [start, end]."""
        return self.start <= d <= self.end

    def overlaps_or_adjacent(self, other: DateInterval) -> bool:
        """Check if this interval overlaps with or is directly adjacent (<= 1 day gap) to another."""
        one_day = timedelta(days=1)
        return not (self.end + one_day < other.start or other.end + one_day < self.start)

    def union(self, other: DateInterval) -> DateInterval:
        """Merge two overlapping or adjacent intervals into one."""
        if not self.overlaps_or_adjacent(other):
            raise ValueError(f"Cannot union non-overlapping/non-adjacent intervals: {self} and {other}")
        return DateInterval(min(self.start, other.start), max(self.end, other.end))

    def intersection(self, other: DateInterval) -> DateInterval | None:
        """Return the intersection interval if they overlap, else None."""
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if start <= end:
            return DateInterval(start, end)
        return None

    def to_dict(self) -> dict[str, str]:
        """Convert to JSON-serializable dict."""
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> DateInterval:
        """Construct from dictionary with 'start' and 'end' ISO strings."""
        return cls(
            start=date.fromisoformat(data["start"]),
            end=date.fromisoformat(data["end"]),
        )


def merge_intervals(intervals: Sequence[DateInterval]) -> list[DateInterval]:
    """Sort and merge overlapping or adjacent date intervals into minimal disjoint intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda iv: iv.start)
    merged: list[DateInterval] = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last = merged[-1]
        if last.overlaps_or_adjacent(current):
            merged[-1] = last.union(current)
        else:
            merged.append(current)

    return merged


def subtract_intervals(
    target: DateInterval,
    covered: Sequence[DateInterval],
) -> list[DateInterval]:
    """Compute the missing gaps (sub-intervals) in target not covered by the given covered intervals.

    Returns:
        List of disjoint DateIntervals representing the missing segments in ascending order.
    """
    merged_covered = merge_intervals(covered)
    gaps: list[DateInterval] = []
    current_start = target.start
    one_day = timedelta(days=1)

    for iv in merged_covered:
        if iv.end < current_start:
            continue
        if iv.start > target.end:
            break

        # If there is a gap before this covered interval
        if iv.start > current_start:
            gap_end = min(iv.start - one_day, target.end)
            if current_start <= gap_end:
                gaps.append(DateInterval(current_start, gap_end))

        # Advance current_start past this covered interval
        if iv.end >= current_start:
            current_start = iv.end + one_day

        if current_start > target.end:
            break

    if current_start <= target.end:
        gaps.append(DateInterval(current_start, target.end))

    return gaps


class IntervalRegistry:
    """Manages covered date intervals per symbol and persists to disk."""

    def __init__(self, registry_file: Path | str | None = None) -> None:
        self.registry_file = Path(registry_file) if registry_file is not None else None
        self._symbol_intervals: dict[str, list[DateInterval]] = {}
        if self.registry_file and self.registry_file.exists():
            self.load()

    def get_intervals(self, symbol: str) -> list[DateInterval]:
        """Get the merged list of covered intervals for a symbol."""
        return list(self._symbol_intervals.get(symbol, []))

    def add_interval(self, symbol: str, interval: DateInterval) -> None:
        """Register a new interval for a symbol and re-merge."""
        existing = self._symbol_intervals.get(symbol, [])
        self._symbol_intervals[symbol] = merge_intervals([*existing, interval])

    def get_missing_gaps(self, symbol: str, target: DateInterval) -> list[DateInterval]:
        """Calculate missing date ranges for a symbol given a target interval."""
        covered = self._symbol_intervals.get(symbol, [])
        return subtract_intervals(target, covered)

    def load(self) -> None:
        """Load registry from JSON file."""
        if not self.registry_file or not self.registry_file.exists():
            return
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._symbol_intervals = {
                symbol: [DateInterval.from_dict(iv) for iv in interval_list]
                for symbol, interval_list in data.items()
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            self._symbol_intervals = {}

    def save(self) -> None:
        """Persist registry to JSON file."""
        if not self.registry_file:
            return
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            symbol: [iv.to_dict() for iv in intervals]
            for symbol, intervals in self._symbol_intervals.items()
        }
        temp_file = self.registry_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.registry_file)
