# Nifty-Quant

[![Python 3.14+](https://img.shields.io/badge/Python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A systematic, **multi-strategy** backtesting framework for the NSE NIFTY 50 universe, built in Python. Strategies are plug-in modules selected at runtime via Hydra config or the web UI — no code changes required to switch between them.

---

## Features

- **Strategy Plugin System** — each strategy lives in its own file; swap with a single CLI flag or dropdown selector.
- **Low-Volatility Anomaly** — ranks universe constituents by trailing realized volatility and weights top-K lowest-risk stocks.
- **Momentum 12-1** — classic cross-sectional momentum with inverse-vol sizing and recent-month skip.
- **AR(p) / ARIMA Signal** — vectorised autoregressive return forecast (batched numpy OLS, ~20,000× faster than MLE).
- **Interval-Aware Local Data Cache** — tracks downloaded date intervals per ticker, calculates and fetches only missing date gaps, and stores unified per-symbol price files without redundant network calls.
- **Async Compute & Job Queue** — background thread execution (`JobManager`) decoupled from web request loops.
- **Real-Time Progress Streaming** — WebSockets (`/ws/backtest/{job_id}`) and REST API (`/api/backtest/...`) stream live backtest progress bars & status updates.
- **Inverse Volatility Sizing** — positions balanced by 60-day rolling σ, capped at 20% per stock with volatility targeting.
- **Realistic Cost Modelling** — brokerage + slippage applied at every rebalance step.
- **NIFTY 50 Universe** — India's 50 largest listed companies.
- **Hydra Configuration** — every parameter (strategy, dates, capital, costs) overrideable from CLI or web UI.
- **Web Dashboard** — FastAPI + Jinja2 + WebSocket interface with dynamic strategy selector dropdown and dedicated parameter subwindows.

---

## Backtest Results (Jan 2022 – Jun 2026)

| Metric | Low Volatility | Momentum 12-1 | AR(2) OLS | AR(2) MLE |
|---|---|---|---|---|
| Total return | +63.85% | **+79.67%** | +58.26% | +55.99% |
| Annual return | **11.90%** | 11.26% | 8.72% | 8.43% |
| Annual volatility | **8.52%** | 10.06% | 10.73% | 10.59% |
| Sharpe ratio | **1.40** | 1.11 | 0.83 | 0.82 |
| Sortino ratio | **1.68** | 1.31 | 1.11 | 1.09 |
| Max drawdown | **−11.98%** | −13.00% | −20.47% | −20.38% |
| Calmar ratio | **0.99** | 0.87 | 0.43 | 0.41 |
| Observations | 1,384 days | 1,384 days | 1,384 days | 1,384 days |
| Time taken | **0.04s** | 30.64s | 135.47s | 282.17s |

> Results include brokerage and slippage costs. Not financial advice.

---

## Model Fitting Micro-Benchmark

Comparing AR(2) model estimation speed across 50 NIFTY constituent stocks (60 trading days matrix per step):

| Model Fitting Engine | Vectorization / Concurrency | Avg Time / Step (50 stocks) | Avg Time / Stock | Speedup |
|---|---|---|---|---|
| **Batched OLS** | SIMD vectorised `numpy.einsum` | **0.184 ms** (0.000184 s) | **0.0037 ms** | **20,726.0× FASTER** |
| **Parallel MLE** | `statsmodels.tsa.arima` (4 Threads) | 3,823.92 ms (3.8239 s) | 76.48 ms | 1.0× (baseline) |
| **Sequential MLE** | `statsmodels.tsa.arima` (Single Thread) | 3,608.14 ms (3.6081 s) | 72.16 ms | 1.06× |

Run the micro-benchmark locally:

```bash
uv run python benchmark_fit_time.py
```

---

## Strategies

### `low_vol`

Ranks all NIFTY 50 stocks by their trailing realized return volatility, selecting the top-10 lowest-volatility stocks to capture the Low-Volatility Anomaly. Positions are weighted inversely to realized volatility.

```yaml
# conf/strategy/low_vol.yaml
name: low_vol
lookback_days: 252      # Trailing days to measure return volatility
top_k: 10
vol_lookback_days: 60
max_weight: 0.20
cash_buffer: 0.05
target_annual_vol: 0.10
```

### `momentum_12_1`

Ranks all NIFTY 50 stocks by their 12-month total return, skipping the most-recent month to avoid short-term reversal. Selects top-10 and weights inversely to realised volatility.

```yaml
# conf/strategy/momentum_12_1.yaml
name: momentum_12_1
lookback_days: 252      # 12-month return window
skip_recent_days: 21    # skip last month (reversal avoidance)
top_k: 10
vol_lookback_days: 60
max_weight: 0.20
cash_buffer: 0.05
target_annual_vol: 0.10
```

### `arima` (default)

Fits an AR(p) model on each stock's log-return history, takes a one-step-ahead forecast, and selects the top-k stocks with the highest **positive** forecast.

Two fitting methods:

| `method` | Engine | Speed | Notes |
|---|---|---|---|
| `ols` (default) | Batched numpy einsum + solve | ~0.18 ms / step (~20,000× faster) | Pure AR(p); no MA terms |
| `mle` | statsmodels ARIMA(p,d,q) | ~3.8 s / step | Full ARIMA with MA terms |

```yaml
# conf/strategy/arima.yaml
name: arima
method: ols        # ols (fast) or mle (statsmodels, slow)
arima_p: 2         # AR lag order
arima_d: 0         # integration order (mle only)
arima_q: 0         # MA order (mle only)
fit_window: 60     # trading days of history fed to model
top_k: 10
vol_lookback_days: 60
max_weight: 0.20
cash_buffer: 0.05
target_annual_vol: 0.10
```

---

## Quick Start

**Requirements:** Python 3.14+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/your-username/Nifty-Quant.git
cd Nifty-Quant
uv sync
```

### Run a backtest

```bash
# Run Low-Volatility strategy
uv run python main.py strategy=low_vol

# Switch to Momentum 12-1
uv run python main.py strategy=momentum_12_1

# Run ARIMA with custom params
uv run python main.py strategy=arima strategy.arima_p=3 strategy.top_k=5

# Override dates and capital (any strategy)
uv run python main.py backtest.start_date=2020-01-01 backtest.initial_capital=500000
```

### Launch the web dashboard

```bash
uv run uvicorn nifty_quant.web.app:app --reload
```

Navigate to `http://127.0.0.1:8000`.

---

## Project Structure

```
Nifty-Quant/
├── conf/                              # Hydra config files
│   ├── config.yaml                    # Root config (assembles all modules)
│   ├── backtest/monthly.yaml          # Date range and capital
│   ├── data/yahoo.yaml                # Data provider
│   ├── execution/india_equities.yaml  # Brokerage + slippage
│   ├── portfolio/inverse_vol.yaml     # Sizing parameters
│   ├── strategy/
│   │   ├── arima.yaml                 # AR/ARIMA strategy config
│   │   ├── low_vol.yaml               # Low-Volatility strategy config
│   │   └── momentum_12_1.yaml         # Momentum strategy config
│   └── universe/nifty50.yaml          # NIFTY 50 symbol list
├── nifty_quant/
│   ├── main.py                        # CLI entry point
│   ├── application/
│   │   ├── backtest_runner.py         # Orchestration layer
│   │   └── job_manager.py             # Async job queue & worker thread pool
│   ├── bootstrap/config_schema.py     # Config validation
│   ├── domain/
│   │   ├── backtest/engine.py         # Strategy-agnostic engine with progress callbacks
│   │   ├── strategies/
│   │   │   ├── base.py                # Strategy ABC
│   │   │   ├── arima.py               # AR/ARIMA implementation
│   │   │   ├── low_vol.py             # Low-Volatility implementation
│   │   │   ├── momentum_12_1.py       # Momentum 12-1 implementation
│   │   │   └── registry.py            # Factory decorator: name → Strategy instance
│   │   ├── metrics.py                 # Performance metrics
│   │   └── models.py                  # BacktestResult dataclass
│   ├── infrastructure/
│   │   ├── data/
│   │   │   ├── cached_price_repository.py # Interval-aware caching decorator
│   │   │   ├── date_intervals.py          # Date interval math & registry
│   │   │   ├── local_price_store.py       # Consolidated per-symbol storage manager
│   │   │   └── yahoo_price_repository.py  # Upstream Yahoo Finance client
│   │   └── execution/india_equities.py
│   ├── interfaces/                    # Abstract interfaces (DI boundaries)
│   └── web/app.py                     # FastAPI dashboard, REST & WebSocket API
├── data/                              # Local data storage (gitignored)
│   └── cache/
│       ├── intervals.json             # Registry of downloaded date ranges per symbol
│       └── prices/*.csv               # Consolidated per-symbol price series
├── templates/                         # HTML Jinja2 dashboard templates
├── static/                            # Dashboard CSS styles and generated assets
├── nifty_ticker/                      # NSE constituent scraper
└── tests/                             # Unit, integration & async job tests
```

---

## Adding a New Strategy

1. Create `nifty_quant/domain/strategies/my_strategy.py` implementing the `Strategy` ABC and decorating with `@register`:

```python
from nifty_quant.domain.strategies.base import Strategy
from nifty_quant.domain.strategies.registry import register

@register("my_strategy")
class MyStrategy(Strategy):
    signal_label: str = "MY SIGNAL"
    rank_note: str = "Ranked by custom signal score."

    @property
    def min_history_days(self) -> int: ...

    def select_and_weight(self, prices, daily_returns, as_of) -> pd.Series: ...

    def compute_signals(self, prices, daily_returns, as_of) -> dict[str, float]: ...
```

2. Add `conf/strategy/my_strategy.yaml`:

```yaml
name: my_strategy
# ... your params
```

3. Run:

```bash
uv run python main.py strategy=my_strategy
```

---

## Configuration Reference

### `conf/config.yaml` — active strategy and module selection

```yaml
defaults:
  - strategy: low_vol   # ← change to: momentum_12_1 or arima
  - portfolio: inverse_vol
  - backtest: monthly
  - universe: nifty50
  - data: yahoo
  - execution: india_equities
```

### `conf/backtest/monthly.yaml`

```yaml
frequency: monthly
start_date: "2022-01-01"
end_date: null           # null = run to today
initial_capital: 1_000_000
rebalance_every: 21      # trading days between rebalances
```

### `conf/data/yahoo.yaml` — local caching and data provider settings

```yaml
provider: yahoo
adjusted_prices: true
use_cache: true          # enable local interval-aware caching
cache_dir: "data/cache"  # directory for cached prices & interval registry
force_refresh: false     # set true to bypass cache and re-download
```

---

## Local Data Store & Interval Caching

To avoid repeatedly re-downloading market data from Yahoo Finance on every backtest run, the framework includes an **Interval-Aware Caching Layer** (`CachedPriceRepository`):

### How It Works

1. **Interval Registry (`intervals.json`)**: Tracks the exact date intervals `[start, end]` downloaded for each symbol.
2. **Smart Gap Detection**: When querying a date range $[R_{start}, R_{end}]$, it computes the set difference against covered intervals ($[R_{start}, R_{end}] \setminus \bigcup \mathcal{I}_{covered}$) to find only missing sub-intervals.
3. **Targeted Upstream Downloads**: Only missing gaps are fetched from Yahoo Finance. Identical gaps across multiple tickers are batched into a single upstream request.
4. **Consolidated Per-Symbol Storage**: Avoids fragmented per-interval files. Each ticker has a single continuous time-series (`data/cache/prices/<SYMBOL>.csv` or `.parquet`) with atomic writes and date deduplication (`keep="last"`).
5. **Seamless Range Merging**: Newly downloaded intervals are merged with existing overlapping or adjacent date ranges.

```bash
# Bypass cache for a single run
uv run python main.py data.force_refresh=true

# Disable caching completely
uv run python main.py data.use_cache=false
```

---

## Architecture

```
Web Dashboard (FastAPI / WebSockets) / CLI
    ├── POST /api/backtest/run
    │      └── JobManager.submit_job() [ThreadPoolExecutor]
    │              └── build_backtest_snapshot(DictConfig, progress_callback)
    │                      ├── build_strategy(cfg)       ← registry factory
    │                      │       └── Strategy.select_and_weight()
    │                      └── BacktestEngine.run(progress_callback)
    │                              ├── CachedPriceRepository  ← interval math & local cache
    │                              │       ├── IntervalRegistry (intervals.json)
    │                              │       ├── LocalPriceStore  (prices/*.csv)
    │                              │       └── YahooPriceRepository (upstream fallback)
    │                              └── ExecutionModel         ← transaction costs
    └── WebSocket /ws/backtest/{job_id} ← streams real-time status & progress %
```

The engine is **strategy-agnostic** — it schedules rebalances, applies costs, and builds the equity curve. All signal generation and sizing live inside the injected `Strategy` object, while compute runs asynchronously off the main ASGI event loop.

---

## Updating the NIFTY 50 Universe

```bash
cd nifty_ticker
python ticket_extractor.py
```

Outputs a timestamped snapshot to `nifty_ticker/nifty_snapshots/`. Copy the symbols into `conf/universe/nifty50.yaml`. Uses `curl_cffi` to replicate Chrome's TLS fingerprint and bypass NSE bot protection.

---

## Running Tests & Benchmarks

```bash
# Run unit and integration tests
uv run pytest tests/ -v

# Run AR model fitting micro-benchmark (Batched OLS vs statsmodels MLE)
uv run python benchmark_fit_time.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Historical price data |
| `pandas` | Data manipulation |
| `numpy` | Numerical computations + vectorised OLS |
| `statsmodels` | ARIMA MLE (optional, `method=mle` only) |
| `hydra-core` | Config management |
| `omegaconf` | Config object model |
| `fastapi` / `uvicorn` | Web dashboard & REST/WebSocket server |
| `httpx` | Async HTTP client for API testing |
| `curl_cffi` | NSE scraper (Chrome TLS impersonation) |

---

## Known Limitations

- **Survivorship bias** — uses today's NIFTY 50 constituents; historical additions/removals are not modelled.
- **Data quality** — relies on Yahoo Finance, which occasionally has gaps or adjusted-price errors.
- **Transaction costs** — computed on initial capital, not current portfolio value (understates costs as equity grows).

---

## Disclaimer

For educational and research purposes only. Past backtest performance is not indicative of future results. Not financial advice.
