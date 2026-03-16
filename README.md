# Nifty-Quant

A systematic momentum backtesting framework for the NSE NIFTY 50 universe, built in Python. Implements a 12-1 momentum strategy with inverse-volatility position sizing, monthly rebalancing, and realistic transaction cost modelling.

---

## Strategy

**Signal — 12-1 Momentum**
At each rebalance date, rank all NIFTY 50 constituents by their 12-month total return ending 1 month ago (252 trading days lookback, skipping the most recent 21 days to avoid short-term reversal). Select the top 10 stocks.

**Sizing — Inverse Volatility**
Weight the 10 selected stocks proportional to `1 / σ` where σ is each stock's 60-day rolling daily return standard deviation. Weights are:

- Capped at 10% per stock (iteratively renormalised)
- Scaled by `1 - cash_buffer` (5% cash held at all times)
- Scaled down if portfolio annualised vol exceeds 10% target (never levered up)

**Rebalance cadence** — every ~21 trading days (monthly)

**Benchmark** — NIFTY 50 buy-and-hold (long-only, no costs)

---

## Backtest Results

| Metric             | Value               |
| ------------------ | ------------------- |
| Period             | Jan 2018 – Mar 2026 |
| Total return       | +159.28%            |
| NIFTY 50 benchmark | ~+85% (approx.)     |
| Initial capital    | ₹10,00,000          |
| Final equity       | ₹25,92,800          |
| Observations       | 2,316 trading days  |
| Rebalances         | 99                  |

> **Note:** Results include brokerage and slippage costs. Survivorship bias is a known limitation — see [Known Issues](#known-issues).

---

## Project Structure

```
Nifty-Quant/
├── conf/                          # Hydra config files
│   ├── config.yaml                # Root config (assembles all modules)
│   ├── backtest/monthly.yaml      # Backtest date range and capital
│   ├── data/yahoo.yaml            # Data provider settings
│   ├── execution/india_equities.yaml  # Brokerage and slippage rates
│   ├── portfolio/inverse_vol.yaml # Position sizing parameters
│   ├── strategy/momentum_12_1.yaml    # Signal parameters
│   └── universe/nifty50.yaml      # NIFTY 50 symbol list
├── nifty_quant/                   # Main package
│   ├── main.py                    # Entry point (Hydra)
│   ├── bootstrap/config_schema.py # Typed config validation
│   ├── domain/
│   │   ├── models.py              # BacktestResult dataclass
│   │   └── backtest/engine.py     # Core backtesting engine
│   ├── infrastructure/
│   │   ├── data/yahoo_price_repository.py   # Yahoo Finance data feed
│   │   └── execution/india_equities.py      # Cost model
│   └── interfaces/
│       ├── execution_model.py     # Abstract execution interface
│       └── price_repository.py    # Abstract price interface
├── nifty_ticker/                  # NSE snapshot scraper
│   └── ticket_extractor.py        # Fetches live NIFTY 50 constituents from NSE
└── tests/                         # Unit and integration tests
```

---

## Installation

**Requirements:** Python 3.11+

```bash
git clone https://github.com/your-username/Nifty-Quant.git
cd Nifty-Quant
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

---

## Running the Backtest

```bash
python -m nifty_quant.main
```

Hydra will load all configs from `conf/` automatically. Output is printed to stdout:

```
Backtest completed
Start equity: 1,000,000.00
End equity:   2,592,800.65
Total return: 159.28%
Observations: 2316
```

**Override any config parameter from the command line:**

```bash
# Change start date
python -m nifty_quant.main backtest.start_date=2020-01-01

# Change top_k
python -m nifty_quant.main strategy.top_k=5

# Change initial capital
python -m nifty_quant.main backtest.initial_capital=500000
```

---

## Configuration

### Strategy (`conf/strategy/momentum_12_1.yaml`)

```yaml
name: momentum_12_1
lookback_days: 252 # 12-month return window
skip_recent_days: 21 # Skip last 1 month (avoids short-term reversal)
top_k: 10 # Number of stocks to hold
```

### Portfolio (`conf/portfolio/inverse_vol.yaml`)

```yaml
method: inverse_volatility
vol_lookback_days: 60 # Rolling window for volatility estimate
max_weight: 0.10 # Maximum weight per stock (10%)
cash_buffer: 0.05 # Cash held at all times (5%)
target_annual_vol: 0.10 # Portfolio vol target (10% p.a.)
```

### Execution (`conf/execution/india_equities.yaml`)

Brokerage and slippage rates for NSE equity trading. Adjust to match your broker.

### Backtest (`conf/backtest/monthly.yaml`)

```yaml
frequency: daily
start_date: "2018-01-01"
end_date: null # null = run to latest available data
initial_capital: 1_000_000
```

---

## Updating the NIFTY 50 Universe

The `nifty_ticker/` module scrapes the live NIFTY 50 constituent list directly from NSE and saves it as both CSV and YAML:

```bash
cd nifty_ticker
python ticket_extractor.py
```

This outputs files to `nifty_ticker/nifty_snapshots/` with a timestamp. Copy the symbols from the generated YAML into `conf/universe/nifty50.yaml` to update the universe.

> Uses `curl_cffi` to replicate Chrome's TLS fingerprint and bypass NSE's bot protection.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Known Issues

**Survivorship bias** — `_align_prices` currently uses `dropna(how="any")`, which silently restricts the price matrix to dates where all 50 symbols have data simultaneously. Symbols with recent listing dates (e.g. Jio Financial — listed 2023) cause the entire dataset to be trimmed to their listing date unless the `thresh` fix is applied. This inflates backtest returns by excluding stocks that were delisted or replaced due to poor performance.

**Fix in progress:** Replace `dropna(how="any")` with a symbol-level filter that drops stocks missing more than 5% of dates, then forward-fills remaining gaps.

**Transaction costs on initial capital** — Costs are currently computed as a fraction of `initial_capital` rather than current portfolio value. This understates costs as the portfolio grows. A fix to use the rolling equity curve value is planned.

---

## Architecture

The engine follows a clean layered architecture with no I/O or side effects in the domain layer:

```
main.py (Hydra entry point)
    └── BacktestEngine (pure domain logic)
            ├── PriceRepository (interface)
            │       └── YahooPriceRepository (infrastructure)
            └── ExecutionModel (interface)
                    └── IndiaEquitiesExecutionModel (infrastructure)
```

This makes the engine fully unit-testable with fake repositories (see `tests/domain/fakes.py`) and easy to swap data providers or execution models without touching strategy logic.

---

## Dependencies

| Package      | Purpose                                  |
| ------------ | ---------------------------------------- |
| `yfinance`   | Historical price data from Yahoo Finance |
| `pandas`     | Data manipulation and time series        |
| `numpy`      | Numerical computations                   |
| `hydra-core` | Config management                        |
| `omegaconf`  | Config object model                      |
| `curl_cffi`  | NSE scraper (Chrome TLS impersonation)   |
| `pyyaml`     | YAML serialisation                       |

---

## Disclaimer

This project is for research and educational purposes only. Past backtest performance is not indicative of future results. This is not financial advice. Always consult a qualified financial advisor before making investment decisions.
