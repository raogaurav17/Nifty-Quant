"""Entry point for the Nifty-Quant investment advisor.

Runs the configured strategy on the current NIFTY 50 universe and prints
ranked buy recommendations. All parameters (strategy, universe, dates,
capital) are driven by the Hydra config under conf/. Pass Hydra dotlist
overrides as CLI arguments, e.g.::

    python main.py strategy=arima backtest.start_date=2023-01-01

The ranking is driven by each strategy's signal score (the actual selection signal).
"""

from __future__ import annotations

from datetime import date
import sys
import time

from dateutil.relativedelta import relativedelta
from omegaconf import OmegaConf
import pandas as pd

from nifty_quant.application.backtest_runner import build_backtest_snapshot, load_config
from nifty_quant.domain.strategies.registry import build_strategy
from nifty_quant.infrastructure.data.yahoo_price_repository import YahooPriceRepository


# ── Display helpers ───────────────────────────────────────────────────────────

def _bar(value: float, max_value: float, width: int = 16) -> str:
    """Render an ASCII bar scaled relative to the max value in the column."""
    if max_value <= 0:
        return "-" * width
    filled = max(0, min(round((value / max_value) * width), width))
    return "#" * filled + "-" * (width - filled)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Execute the stock advisor workflow and print ranked recommendations."""
    # Read Hydra dotlist overrides straight from the command line.
    # Always run to today unless the caller passes backtest.end_date explicitly.
    overrides = sys.argv[1:]
    if not any(o.startswith("backtest.end_date=") for o in overrides):
        overrides = list(overrides) + ["backtest.end_date=null"]

    # Load config once; pass the DictConfig object through to avoid a second
    # Hydra initialisation inside build_backtest_snapshot.
    cfg = load_config(overrides)
    config = OmegaConf.to_container(cfg, resolve=True)

    today = date.today()
    print()
    print("=" * 72)
    print("  NIFTY-QUANT  --  STOCK ADVISOR")
    print(f"  As of: {today}")
    print("=" * 72)
    print()
    print("Fetching latest price data and running strategy...")
    print()

    t0 = time.perf_counter()
    snapshot = build_backtest_snapshot(cfg)
    elapsed = time.perf_counter() - t0

    holdings = snapshot.holdings  # list[dict], sorted by weight descending

    if not holdings:
        print("WARNING: No holdings generated -- check date range or data source.")
        return

    # ── Read strategy params from Hydra config ────────────────────────────────

    strategy_cfg = config.get("strategy", {})
    strategy_name = str(strategy_cfg.get("name", "momentum_12_1"))

    universe_cfg = config.get("universe", {})
    backtest_cfg = config.get("backtest", {})
    initial_capital = float(backtest_cfg.get("initial_capital", 1_000_000))

    selected_symbols = [h["symbol"] for h in holdings]

    # ── Build the ranked table, strategy-agnostic & versatile ────────────────

    strategy = snapshot.strategy or build_strategy(strategy_cfg)

    fetch_from = today - relativedelta(months=14)
    repo = YahooPriceRepository()
    price_data = repo.get_prices(
        symbols=selected_symbols, start_date=fetch_from, end_date=today
    )
    aligned = [df["adj_close"].rename(sym) for sym, df in price_data.items() if "adj_close" in df.columns]
    prices = pd.concat(aligned, axis=1).ffill() if aligned else pd.DataFrame()
    daily_returns = prices.pct_change().dropna()

    as_of_ts = pd.Timestamp(today)
    try:
        signal_scores = strategy.compute_signals(
            prices=prices,
            daily_returns=daily_returns,
            as_of=as_of_ts,
        )
    except Exception:
        signal_scores = {}

    if signal_scores:
        signal_label = getattr(strategy, "signal_label", "SIGNAL SCORE")
        rank_note = getattr(strategy, "rank_note", f"Ranked by {strategy_name} signal score.")
        holdings_sorted = sorted(
            holdings,
            key=lambda h: signal_scores.get(h["symbol"], float("-inf")),
            reverse=True,
        )
        has_signal_scores = True
    else:
        signal_scores = {}
        signal_label = "WEIGHT"
        rank_note = f"Ranked by portfolio weight -- reflects the {strategy_name} signal."
        holdings_sorted = holdings  # already sorted by weight descending
        has_signal_scores = False

    max_score = max(signal_scores.values()) if signal_scores else 1.0

    # ── Buy recommendations table ─────────────────────────────────────────────

    print("=" * 72)
    print(f"  BUY RECOMMENDATIONS  (strategy: {strategy_name})")
    print(f"  {rank_note}")
    print("=" * 72)

    if has_signal_scores:
        print(
            f"  {'RANK':<5} {'SYMBOL':<14} {signal_label:>11}  "
            f"{'BAR':<18} {'WEIGHT':>7}"
        )
        print("-" * 72)
        for rank, h in enumerate(holdings_sorted, start=1):
            sym = h["symbol"]
            weight = h["percent"]
            score = signal_scores.get(sym)
            if score is not None:
                score_str = f"{score:>+.2%}"
                bar = _bar(score, max_score)
            else:
                score_str = "  n/a"
                bar = "-" * 16
            print(f"  {rank:<5} {sym:<14} {score_str:>11}  {bar:<18} {weight:>6.2f}%")
    else:
        max_weight = max(h["percent"] for h in holdings_sorted) if holdings_sorted else 1.0
        print(f"  {'RANK':<5} {'SYMBOL':<14} {'BAR':<20} {'WEIGHT':>7}")
        print("-" * 52)
        for rank, h in enumerate(holdings_sorted, start=1):
            sym = h["symbol"]
            weight = h["percent"]
            bar = _bar(weight, max_weight)
            print(f"  {rank:<5} {sym:<14} {bar:<20} {weight:>6.2f}%")

    print("=" * 72)

    # ── Portfolio summary ─────────────────────────────────────────────────────

    total_invested = sum(h["weight"] for h in holdings)

    print()
    print("  PORTFOLIO SUMMARY")
    print("-" * 44)
    print(f"  Strategy          : {strategy_name}")
    print(f"  Universe          : {len(universe_cfg.get('symbols', []))} symbols")
    print(f"  Initial capital   : Rs {initial_capital:,.0f}")
    print(f"  Stocks selected   : {len(holdings)}")
    print(f"  Capital deployed  : {total_invested:.1%}  (remainder is cash buffer + vol scaling)")

    # ── Historical performance -- all available metrics from PerformanceMetrics

    print()
    print("  HISTORICAL PERFORMANCE  (backtest)")
    print("-" * 44)
    m = snapshot.metrics
    print(f"  Total return      : {m.total_return:>+.2%}")
    print(f"  Annual return     : {m.annual_return:>+.2%}")
    print(f"  Annual volatility : {m.volatility_annual:>8.2%}")
    print(f"  Downside vol      : {m.downside_volatility_annual:>8.2%}")
    print(f"  Max drawdown      : {m.max_drawdown:>+.2%}")
    print(f"  Sharpe ratio      : {m.sharpe_ratio:>8.2f}")
    print(f"  Sortino ratio     : {m.sortino_ratio:>8.2f}")
    if m.calmar_ratio is not None:
        print(f"  Calmar ratio      : {m.calmar_ratio:>8.2f}")

    # Backtest period and observation count come directly from the snapshot.
    print()
    period_end = snapshot.end_date if snapshot.end_date else date.today()
    print(f"  Backtest period   : {snapshot.start_date}  ->  {period_end}")
    print(f"  Observations      : {len(snapshot.result.returns):>6,} days")

    # ── Recent rebalances from snapshot.recent_trades ─────────────────────────

    if snapshot.recent_trades:
        print()
        print("  RECENT REBALANCES")
        print("-" * 30)
        print(f"  {'DATE':<12}  {'TURNOVER':>10}")
        print("-" * 30)
        for trade in snapshot.recent_trades:
            print(f"  {trade['date']:<12}  {trade['turnover']:>10.4f}")

    print()
    print(f"  Analysis completed in {elapsed:.1f}s")
    print()
    print("  DISCLAIMER: For educational and research purposes only.")
    print("  Past performance is not indicative of future results.")
    print("  This is not financial advice.")
    print()


if __name__ == "__main__":
    main()

