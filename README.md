# Nifty-Quant

[![Python 3.14+](https://img.shields.io/badge/Python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A systematic, **multi-strategy** backtesting framework for the NSE NIFTY 50 universe, built in Python. Strategies are plug-in modules selected at runtime via Hydra config — no code changes required to switch between them.

---

## Live Demo

**[https://nifty-quant.onrender.com/](https://nifty-quant.onrender.com/)**

---

## Features

- **Strategy Plugin System** — each strategy lives in its own file; swap with a single CLI flag.
- **Momentum 12-1** — classic cross-sectional momentum with inverse-vol sizing.
- **AR(p) / ARIMA Signal** — vectorised autoregressive return forecast (batched numpy OLS, ~1000× faster than MLE).
- **Inverse Volatility Sizing** — positions balanced by 60-day rolling σ, capped at 10% per stock.
- **Realistic Cost Modelling** — brokerage + slippage applied at every rebalance.
- **NIFTY 50 Universe** — India's 50 largest listed companies.
- **Hydra Configuration** — every parameter (strategy, dates, capital, costs) overrideable from the CLI.
- **Web Dashboard** — FastAPI + Jinja2 interface to visualise and replay backtests.

---

## Backtest Results (Jan 2022 – Jun 2026)

| Metric | Momentum 12-1 | AR(2) OLS | AR(2) MLE |
|---|---|---|---|
| Total return | **+79.67%** | +58.26% | +55.99% |
| Annual return | **11.26%** | 8.72% | 8.43% |
| Annual volatility | 10.06% | 10.73% | 10.59% |
| Sharpe ratio | **1.11** | 0.83 | 0.82 |
| Sortino ratio | **1.31** | 1.11 | 1.09 |
| Max drawdown | **−13.00%** | −20.47% | −20.38% |
| Calmar ratio | **0.87** | 0.43 | 0.41 |
| Observations | 1,384 days | 1,384 days | 1,384 days |
| Time taken | 30.64s | 135.47s | 282.17s |

> Results include brokerage and slippage costs. Not financial advice.

---

## Strategies

### `momentum_12_1` (default)

Ranks all NIFTY 50 stocks by their 12-month total return, skipping the most-recent month to avoid short-term reversal. Selects top-10 and weights inversely to realised volatility.

```yaml
# conf/strategy/momentum_12_1.yaml
name: momentum_12_1
lookback_days: 252      # 12-month return window
skip_recent_days: 21    # skip last month (reversal avoidance)
top_k: 10
vol_lookback_days: 60
max_weight: 0.10
cash_buffer: 0.05
target_annual_vol: 0.10
```

### `arima`

Fits an AR(p) model on each stock's log-return history, takes a one-step-ahead forecast, and selects the top-k stocks with the highest **positive** forecast.

Two fitting methods:

| `method` | Engine | Speed | Notes |
|---|---|---|---|
| `ols` (default) | Batched numpy einsum + solve | < 5 ms / rebalance | Pure AR(p); no MA terms |
| `mle` | statsmodels ARIMA(p,d,q) | ~1–3 min / rebalance | Full ARIMA with MA terms |

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
max_weight: 0.10
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
# Default strategy (momentum 12-1)
uv run python -m nifty_quant.main

# Switch to ARIMA
uv run python -m nifty_quant.main strategy=arima

# ARIMA with custom params
uv run python -m nifty_quant.main strategy=arima strategy.arima_p=3 strategy.top_k=5

# Override dates and capital (any strategy)
uv run python -m nifty_quant.main backtest.start_date=2020-01-01 backtest.initial_capital=500000
```

### Launch the web dashboard

```bash
uvicorn nifty_quant.web.app:app --reload
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
│   │   ├── momentum_12_1.yaml         # Momentum strategy config
│   │   └── arima.yaml                 # AR/ARIMA strategy config
│   └── universe/nifty50.yaml          # NIFTY 50 symbol list
├── nifty_quant/
│   ├── main.py                        # CLI entry point
│   ├── application/backtest_runner.py # Orchestration layer
│   ├── bootstrap/config_schema.py     # Config validation
│   ├── domain/
│   │   ├── backtest/engine.py         # Strategy-agnostic backtest engine
│   │   ├── strategies/
│   │   │   ├── base.py                # Strategy ABC
│   │   │   ├── momentum_12_1.py       # Momentum 12-1 implementation
│   │   │   ├── arima.py               # AR/ARIMA implementation
│   │   │   └── registry.py            # Factory: name → Strategy instance
│   │   ├── metrics.py                 # Performance metrics
│   │   └── models.py                  # BacktestResult dataclass
│   ├── infrastructure/
│   │   ├── data/yahoo_price_repository.py
│   │   └── execution/india_equities.py
│   ├── interfaces/                    # Abstract interfaces (DI boundaries)
│   └── web/app.py                     # FastAPI dashboard
├── nifty_ticker/                      # NSE constituent scraper
└── tests/                             # Unit + integration tests
```

---

## Adding a New Strategy

1. Create `nifty_quant/domain/strategies/my_strategy.py` implementing the `Strategy` ABC:

```python
from nifty_quant.domain.strategies.base import Strategy

class MyStrategy(Strategy):
    @property
    def min_history_days(self) -> int: ...

    def select_and_weight(self, prices, daily_returns, as_of) -> pd.Series: ...
```

2. Register it in `registry.py`:

```python
_REGISTRY.setdefault("my_strategy", MyStrategy)
```

3. Add `conf/strategy/my_strategy.yaml`:

```yaml
name: my_strategy
# ... your params
```

4. Run:

```bash
uv run python -m nifty_quant.main strategy=my_strategy
```

---

## Configuration Reference

### `conf/config.yaml` — active strategy and module selection

```yaml
defaults:
  - strategy: momentum_12_1   # ← change to: arima
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

---

## Architecture

```
CLI (argparse overrides)
    └── load_config()           ← Hydra compose API
            └── build_backtest_snapshot(DictConfig)
                    ├── build_strategy(cfg)       ← registry factory
                    │       └── Strategy.select_and_weight()
                    └── BacktestEngine.run()
                            ├── PriceRepository   ← interface
                            └── ExecutionModel    ← interface
```

The engine is **strategy-agnostic** — it schedules rebalances, applies costs, and builds the equity curve. All signal generation and sizing live inside the injected `Strategy` object.

---

## Updating the NIFTY 50 Universe

```bash
cd nifty_ticker
python ticket_extractor.py
```

Outputs a timestamped snapshot to `nifty_ticker/nifty_snapshots/`. Copy the symbols into `conf/universe/nifty50.yaml`. Uses `curl_cffi` to replicate Chrome's TLS fingerprint and bypass NSE bot protection.

---

## Running Tests

```bash
uv run pytest tests/ -v
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
| `fastapi` / `uvicorn` | Web dashboard |
| `curl_cffi` | NSE scraper (Chrome TLS impersonation) |

---

## Known Limitations

- **Survivorship bias** — uses today's NIFTY 50 constituents; historical additions/removals are not modelled.
- **Data quality** — relies on Yahoo Finance, which occasionally has gaps or adjusted-price errors.
- **Transaction costs** — computed on initial capital, not current portfolio value (understates costs as equity grows).

---

## Disclaimer

For educational and research purposes only. Past backtest performance is not indicative of future results. Not financial advice.
