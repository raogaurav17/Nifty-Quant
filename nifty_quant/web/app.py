from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from nifty_quant.application.backtest_runner import build_backtest_snapshot, load_config
from nifty_quant.domain.strategies.registry import available_strategies


BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="Nifty Quant Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Strategy-specific query params
_STRATEGY_PARAMS: dict[str, list[tuple[str, str]]] = {
    "momentum_12_1": [
        ("strategy.lookback_days",    "lookback_days"),
        ("strategy.skip_recent_days", "skip_recent_days"),
    ],
    "arima": [
        ("strategy.arima_p",   "arima_p"),
        ("strategy.fit_window","fit_window"),
        ("strategy.method",    "arima_method"),
    ],
}


def _query_overrides(request: Request) -> list[str]:
    params = request.query_params
    overrides: list[str] = []

    # Strategy group
    strategy = params.get("strategy", "momentum_12_1")
    overrides.append(f"strategy={strategy}")

    # Backtest params
    for key, qp in [
        ("backtest.start_date",      "start_date"),
        ("backtest.end_date",        "end_date"),
        ("backtest.initial_capital", "initial_capital"),
    ]:
        v = params.get(qp)
        if v not in (None, ""):
            overrides.append(f"{key}={v}")

    # Sizing params
    for key, qp in [
        ("strategy.top_k",              "top_k"),
        ("strategy.vol_lookback_days",  "vol_lookback_days"),
        ("strategy.max_weight",         "max_weight"),
        ("strategy.cash_buffer",        "cash_buffer"),
        ("strategy.target_annual_vol",  "target_annual_vol"),
    ]:
        v = params.get(qp)
        if v not in (None, ""):
            overrides.append(f"{key}={v}")

    # Strategy-specific params
    for key, qp in _STRATEGY_PARAMS.get(strategy, []):
        v = params.get(qp)
        if v not in (None, ""):
            overrides.append(f"{key}={v}")

    return overrides


def _query_values(request: Request, base_cfg: dict[str, Any]) -> dict[str, Any]:
    params = request.query_params
    strat = base_cfg.get("strategy", {})
    bt = base_cfg.get("backtest", {})

    def _p(qp: str, fallback: Any) -> Any:
        return params.get(qp) or fallback

    return {
        "strategy": _p("strategy", strat.get("name", "momentum_12_1")),
        "start_date": _p("start_date", bt.get("start_date", "")),
        "end_date": _p("end_date", bt.get("end_date") or ""),
        "initial_capital": _p("initial_capital", bt.get("initial_capital", 1_000_000)),
        "top_k": _p("top_k", strat.get("top_k", 10)),
        "vol_lookback_days": _p("vol_lookback_days", strat.get("vol_lookback_days", 60)),
        "max_weight": _p("max_weight", strat.get("max_weight", 0.10)),
        "cash_buffer": _p("cash_buffer", strat.get("cash_buffer", 0.05)),
        "target_annual_vol": _p("target_annual_vol", strat.get("target_annual_vol", 0.10)),
        "lookback_days": _p("lookback_days", strat.get("lookback_days", 252)),
        "skip_recent_days": _p("skip_recent_days", strat.get("skip_recent_days", 21)),
        "arima_p": _p("arima_p", strat.get("arima_p", 2)),
        "fit_window": _p("fit_window", strat.get("fit_window", 60)),
        "arima_method": _p("arima_method", strat.get("method", "ols")),
    }


def _summary_cards(snapshot) -> list[dict]:
    return [
        {"label": "Total return",   "value": f"{snapshot.metrics.total_return:.2%}"},
        {"label": "Annual return",  "value": f"{snapshot.metrics.annual_return:.2%}"},
        {"label": "Volatility",     "value": f"{snapshot.metrics.volatility_annual:.2%}"},
        {"label": "Sharpe",         "value": f"{snapshot.metrics.sharpe_ratio:.2f}"},
        {"label": "Max drawdown",   "value": f"{snapshot.metrics.max_drawdown:.2%}"},
        {"label": "Days traded",    "value": f"{len(snapshot.result.returns):,}"},
    ]


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    base_cfg = load_config([])
    form_values = _query_values(request, base_cfg)
    should_run = request.query_params.get("run") == "1"

    snapshot = None
    error_message = None
    if should_run:
        try:
            snapshot = build_backtest_snapshot(_query_overrides(request))
        except Exception as exc:
            error_message = str(exc)

    context = {
        "request": request,
        "base_config": base_cfg,
        "form_values": form_values,
        "strategies": available_strategies(),
        "snapshot": snapshot,
        "summary_cards": _summary_cards(snapshot) if snapshot else [],
        "error_message": error_message,
        "equity_end": snapshot.result.equity_curve.iloc[-1] if snapshot else None,
        "chart_path": snapshot.chart_path if snapshot else "",
        "chart_min": snapshot.chart_min if snapshot else 0.0,
        "chart_max": snapshot.chart_max if snapshot else 0.0,
        "holdings": snapshot.holdings if snapshot else [],
    }
    return templates.TemplateResponse(request, "dashboard.html", context)
