import time
from curl_cffi import requests
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("nifty_snapshots")
OUTPUT_DIR.mkdir(exist_ok=True)

NSE_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"


def fetch_nifty50():
    session = requests.Session(impersonate="chrome120")

    # Seed session cookies exactly as a real browser would
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

    # Keep only equity-series rows (drops the index summary row)
    df = df[df["series"] == "EQ"].reset_index(drop=True)

    symbols = df["symbol"].tolist()

    if len(symbols) != 50:
        raise ValueError(f"Expected 50 constituents, got {len(symbols)}")

    yahoo_symbols = [s + ".NS" for s in symbols]

    return df, yahoo_symbols


def save_snapshot(df, yahoo_symbols):
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    csv_path = OUTPUT_DIR / f"nifty50_{ts}.csv"
    yaml_path = OUTPUT_DIR / f"nifty50_{ts}.yaml"

    df.to_csv(csv_path, index=False)

    yaml_data = {
        "name": "nifty50",
        "timestamp": ts,
        "symbols": yahoo_symbols
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