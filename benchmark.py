"""
benchmark.py
------------
Computes NIFTY 50 buy-and-hold benchmark metrics over the same period
as the Nifty-Quant backtest, for direct strategy comparison.

Usage:
    python benchmark.py
    python benchmark.py --start 2018-01-01 --end 2026-03-14
    python benchmark.py --start 2020-01-01 --capital 500000
"""

import argparse
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

NIFTY_TICKER   = "^NSEI"     # NIFTY 50 price index (Yahoo Finance)
TRADING_DAYS   = 252         # annualisation factor


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def fetch_index(start: str, end: str | None) -> pd.Series:
    raw = yf.download(
        NIFTY_TICKER,
        start=start,
        end=end,
        auto_adjust=True,      # adjusted close; includes dividend effect
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("No data returned for ^NSEI. Check your internet connection.")

    close = raw["Close"].squeeze()
    close.name = "NIFTY50"
    return close.dropna()


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(prices: pd.Series, initial_capital: float) -> dict:
    returns       = prices.pct_change().dropna()
    equity_curve  = (1 + returns).cumprod() * initial_capital

    total_days    = (prices.index[-1] - prices.index[0]).days
    years         = total_days / 365.25

    total_return  = (prices.iloc[-1] / prices.iloc[0]) - 1
    cagr          = (1 + total_return) ** (1 / years) - 1

    annual_vol    = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    # Max drawdown
    rolling_max   = equity_curve.cummax()
    drawdowns     = (equity_curve - rolling_max) / rolling_max
    max_drawdown  = drawdowns.min()

    # Sharpe (assume 6.5% risk-free rate for India)
    rf_daily      = 0.065 / TRADING_DAYS
    excess        = returns - rf_daily
    sharpe        = (excess.mean() / returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)

    # Calmar
    calmar        = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        "start_date"     : prices.index[0].date(),
        "end_date"       : prices.index[-1].date(),
        "years"          : round(years, 2),
        "start_value"    : round(float(prices.iloc[0]), 2),
        "end_value"      : round(float(prices.iloc[-1]), 2),
        "start_equity"   : round(initial_capital, 2),
        "end_equity"     : round(float(equity_curve.iloc[-1]), 2),
        "total_return"   : total_return,
        "cagr"           : cagr,
        "annual_vol"     : annual_vol,
        "max_drawdown"   : max_drawdown,
        "sharpe"         : sharpe,
        "calmar"         : calmar,
        "observations"   : len(returns),
        "equity_curve"   : equity_curve,
        "returns"        : returns,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────────

def print_report(m: dict, strategy_return: float | None = None) -> None:
    SEP = "─" * 45

    print(f"\n{'NIFTY 50 BENCHMARK REPORT':^45}")
    print(SEP)
    print(f"  Period          {m['start_date']}  →  {m['end_date']}")
    print(f"  Duration        {m['years']:.1f} years  ({m['observations']:,} trading days)")
    print(SEP)
    print(f"  Index start     {m['start_value']:>12,.2f}")
    print(f"  Index end       {m['end_value']:>12,.2f}")
    print(SEP)
    print(f"  Start equity    ₹{m['start_equity']:>12,.2f}")
    print(f"  End equity      ₹{m['end_equity']:>12,.2f}")
    print(SEP)
    print(f"  Total return    {m['total_return']:>+11.2%}")
    print(f"  CAGR            {m['cagr']:>+11.2%}  p.a.")
    print(f"  Annual vol      {m['annual_vol']:>11.2%}  p.a.")
    print(f"  Max drawdown    {m['max_drawdown']:>11.2%}")
    print(f"  Sharpe ratio    {m['sharpe']:>11.2f}  (Rf = 6.5%)")
    print(f"  Calmar ratio    {m['calmar']:>11.2f}")

    if strategy_return is not None:
        alpha = strategy_return - m['total_return']
        print(SEP)
        print(f"  Strategy return {strategy_return:>+11.2%}")
        print(f"  Benchmark       {m['total_return']:>+11.2%}")
        print(f"  Alpha           {alpha:>+11.2%}  total")

    print(SEP)

    # Worst years
    annual = (
        m["returns"]
        .groupby(m["returns"].index.year)
        .apply(lambda r: (1 + r).prod() - 1)
    )
    print("\n  Annual returns:")
    for yr, ret in annual.items():
        bar = "█" * int(abs(ret) * 40)
        sign = "+" if ret >= 0 else "-"
        print(f"  {yr}  {sign}{abs(ret):5.1%}  {bar}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="NIFTY 50 benchmark calculator")
    parser.add_argument("--start",    default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      default=None,         help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--capital",  default=1_000_000,    type=float, help="Initial capital in INR")
    parser.add_argument("--strategy", default=None,         type=float,
                        help="Strategy total return as decimal e.g. 1.5928 for 159.28%%")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Fetching NIFTY 50 data from {args.start} to {args.end or 'today'}...")
    prices = fetch_index(start=args.start, end=args.end)

    metrics = compute_metrics(prices, initial_capital=args.capital)
    print_report(metrics, strategy_return=args.strategy)