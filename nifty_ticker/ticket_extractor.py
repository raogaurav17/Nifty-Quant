"""Constituent extractor utility for fetching latest NIFTY 50 tickers from NSE."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

from curl_cffi import requests
import pandas as pd
import yaml

OUTPUT_DIR = Path("nifty_snapshots")
OUTPUT_DIR.mkdir(exist_ok=True)

NSE_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"


def fetch_nifty50() -> tuple[pd.DataFrame, list[str]]:
    """Fetch current NIFTY 50 index constituents from the NSE website.

    Returns:
        Tuple containing constituents DataFrame and Yahoo Finance compatible symbol list.
    """
    session = requests.Session(impersonate="chrome120")

    session.get("https://www.nseindia.com", timeout=15)
    time.sleep(2)
    session.get(
        "https://www.nseindia.com/market-data/live-equity-market",
        timeout=15,
    )
    time.sleep(1)

    response = session.get(NSE_URL, timeout=15)
    response.raise_for_status()

    if not response.text.strip():
        raise ValueError(
            f"NSE returned an empty response (status {response.status_code}). "
            "Try running again — the site may have rate-limited this IP."
        )

    data = response.json()
    df = pd.DataFrame(data["data"])
    df = df[df["series"] == "EQ"].reset_index(drop=True)

    symbols = df["symbol"].tolist()
    if len(symbols) != 50:
        raise ValueError(f"Expected 50 constituents, got {len(symbols)}")

    yahoo_symbols = [s + ".NS" for s in symbols]
    return df, yahoo_symbols


def save_snapshot(df: pd.DataFrame, yahoo_symbols: list[str]) -> None:
    """Save constituent snapshot to CSV and YAML in nifty_snapshots directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    csv_path = OUTPUT_DIR / f"nifty50_{ts}.csv"
    yaml_path = OUTPUT_DIR / f"nifty50_{ts}.yaml"

    df.to_csv(csv_path, index=False)

    yaml_data = {
        "name": "nifty50",
        "timestamp": ts,
        "symbols": yahoo_symbols,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    print(f"Saved CSV → {csv_path}")
    print(f"Saved YAML → {yaml_path}")


if __name__ == "__main__":
    df, yahoo_symbols = fetch_nifty50()

    print("\nLatest NIFTY50 Constituents:\n")
    for s in yahoo_symbols:
        print(s)

    save_snapshot(df, yahoo_symbols)