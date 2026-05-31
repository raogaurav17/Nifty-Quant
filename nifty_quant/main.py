from argparse import ArgumentParser
from datetime import date

from nifty_quant.application.backtest_runner import build_backtest_snapshot


def main() -> None:

    parser = ArgumentParser(description="Run the Nifty-Quant backtest")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist form, for example backtest.start_date=2020-01-01",
    )
    args = parser.parse_args()

    snapshot = build_backtest_snapshot(args.overrides)

    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Start equity:        ₹{snapshot.result.equity_curve.iloc[0]:>15,.2f}")
    print(f"End equity:          ₹{snapshot.result.equity_curve.iloc[-1]:>15,.2f}")
    print()
    print(f"Total return:        {snapshot.metrics.total_return:>15.2%}")
    print(f"Annual return:       {snapshot.metrics.annual_return:>15.2%}")
    print(f"Annual volatility:   {snapshot.metrics.volatility_annual:>15.2%}")
    print()
    print(f"Sharpe ratio:        {snapshot.metrics.sharpe_ratio:>15.4f}")
    print(f"Sortino ratio:       {snapshot.metrics.sortino_ratio:>15.4f}")
    print(f"Max drawdown:        {snapshot.metrics.max_drawdown:>15.2%}")
    if snapshot.metrics.calmar_ratio is not None:
        print(f"Calmar ratio:        {snapshot.metrics.calmar_ratio:>15.4f}")
    print()
    print(f"Observations:        {len(snapshot.result.returns):>15,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
