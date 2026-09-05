"""Time-aware dynamic universe provider for survivorship bias-free backtesting."""

from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nifty_quant.interfaces.universe_provider import UniverseProvider


def _normalize_date(dt: date | datetime | pd.Timestamp | str) -> date:
    """Normalize various date-like representations to a standard date."""
    if isinstance(dt, str):
        return datetime.strptime(dt[:10], "%Y-%m-%d").date()
    if isinstance(dt, pd.Timestamp):
        return dt.date()
    if isinstance(dt, datetime):
        return dt.date()
    return dt


class DynamicUniverseProvider(UniverseProvider):
    """Reconstructs exact index constituents across time without survivorship bias."""

    def __init__(
        self,
        timeline_source: str | Path | dict[str, Any],
        name: str = "nifty50_dynamic",
    ) -> None:
        self._name = name
        if isinstance(timeline_source, (str, Path)):
            path = Path(timeline_source)
            if not path.is_absolute():
                # Allow relative to workspace root
                path = Path.cwd() / path
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = timeline_source

        self._baseline_date = _normalize_date(data.get("baseline_date", "2026-03-15"))
        self._baseline_constituents = set(data.get("baseline_constituents", []))
        if not self._baseline_constituents:
            raise ValueError("Timeline baseline_constituents cannot be empty.")

        raw_events = data.get("reconstitutions", [])
        # Sort events chronologically ascending
        self._events: list[dict[str, Any]] = sorted(
            raw_events,
            key=lambda e: _normalize_date(e["effective_date"]),
        )

        # Build chronological interval periods: list of (effective_date, set_of_constituents)
        self._interval_dates: list[date] = []
        self._interval_constituents: list[list[str]] = []
        self._build_intervals()

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_dynamic(self) -> bool:
        return True

    def _build_intervals(self) -> None:
        """Pre-compute the constituent set for every historical interval."""
        # Start backward from baseline constituents
        # For any event occurring before or at baseline:
        # Step backwards from baseline to determine constituents before each event
        current_set = set(self._baseline_constituents)

        # Separate events into those <= baseline_date and those > baseline_date
        past_events = [e for e in self._events if _normalize_date(e["effective_date"]) <= self._baseline_date]
        future_events = [e for e in self._events if _normalize_date(e["effective_date"]) > self._baseline_date]

        # Milestone states: date -> constituent set starting ON that date
        milestones: dict[date, set[str]] = {}

        # Past events in reverse order: before event E, constituent set had E.removed IN and E.added OUT
        for event in reversed(past_events):
            eff_dt = _normalize_date(event["effective_date"])
            milestones[eff_dt] = set(current_set)
            # Revert changes to find state before eff_dt
            added = set(event.get("added", []))
            removed = set(event.get("removed", []))
            current_set = (current_set - added) | removed

        # State prior to the earliest event
        earliest_date = date.min
        if past_events:
            milestones[earliest_date] = set(current_set)
        else:
            milestones[earliest_date] = set(self._baseline_constituents)

        # Baseline date itself
        milestones[self._baseline_date] = set(self._baseline_constituents)

        # Future events after baseline
        current_fwd = set(self._baseline_constituents)
        for event in future_events:
            eff_dt = _normalize_date(event["effective_date"])
            added = set(event.get("added", []))
            removed = set(event.get("removed", []))
            current_fwd = (current_fwd - removed) | added
            milestones[eff_dt] = set(current_fwd)

        # Sort milestones chronologically
        sorted_dates = sorted(milestones.keys())
        self._interval_dates = sorted_dates
        self._interval_constituents = [sorted(list(milestones[d])) for d in sorted_dates]

    def get_constituents(self, as_of: date | pd.Timestamp) -> list[str]:
        """Return the active universe constituent symbols as of a given date."""
        target_date = _normalize_date(as_of)
        # Find rightmost date <= target_date
        idx = bisect_right(self._interval_dates, target_date) - 1
        if idx < 0:
            idx = 0
        return list(self._interval_constituents[idx])

    def get_all_symbols(
        self,
        start_date: date | pd.Timestamp,
        end_date: date | pd.Timestamp | None = None,
    ) -> list[str]:
        """Return union of all symbols that were active at any point in [start_date, end_date]."""
        start = _normalize_date(start_date)
        end = _normalize_date(end_date) if end_date is not None else date.max

        all_syms: set[str] = set()
        for i, interval_dt in enumerate(self._interval_dates):
            # Check if this interval overlaps [start, end]
            next_dt = self._interval_dates[i + 1] if i + 1 < len(self._interval_dates) else date.max
            if interval_dt <= end and next_dt >= start:
                all_syms.update(self._interval_constituents[i])

        return sorted(list(all_syms))

    def get_events(
        self,
        start_date: date | pd.Timestamp | None = None,
        end_date: date | pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        """Return reconstitution events that occurred within the date interval."""
        start = _normalize_date(start_date) if start_date is not None else date.min
        end = _normalize_date(end_date) if end_date is not None else date.max

        filtered = []
        for e in self._events:
            eff_dt = _normalize_date(e["effective_date"])
            if start <= eff_dt <= end:
                filtered.append(dict(e))
        return filtered
