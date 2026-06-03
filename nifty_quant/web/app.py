from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from nifty_quant.application.backtest_runner import build_backtest_snapshot, load_config


BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="Nifty Quant Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _query_overrides(request: Request) -> list[str]:
    params = request.query_params
    mappings = {
        "backtest.start_date": params.get("start_date"),
        "backtest.end_date": params.get("end_date"),
        "backtest.initial_capital": params.get("initial_capital"),
        "strategy.lookback_days": params.get("lookback_days"),
        "strategy.skip_recent_days": params.get("skip_recent_days"),
        "strategy.top_k": params.get("top_k"),
        "portfolio.vol_lookback_days": params.get("vol_lookback_days"),
        "portfolio.max_weight": params.get("max_weight"),
        "portfolio.cash_buffer": params.get("cash_buffer"),
        "portfolio.target_annual_vol": params.get("target_annual_vol"),
    }

    overrides: list[str] = []
    for key, value in mappings.items():
        if value not in (None, ""):
            overrides.append(f"{key}={value}")
    return overrides


def _query_values(request: Request, base_config: dict) -> dict:
    params = request.query_params
    return {
        "start_date": params.get("start_date") or base_config["backtest"]["start_date"],
        "end_date": params.get("end_date") or (base_config["backtest"].get("end_date") or ""),
        "initial_capital": params.get("initial_capital") or base_config["backtest"]["initial_capital"],
        "lookback_days": params.get("lookback_days") or base_config["strategy"]["lookback_days"],
        "skip_recent_days": params.get("skip_recent_days") or base_config["strategy"]["skip_recent_days"],
        "top_k": params.get("top_k") or base_config["strategy"]["top_k"],
        "vol_lookback_days": params.get("vol_lookback_days") or base_config["portfolio"]["vol_lookback_days"],
        "max_weight": params.get("max_weight") or base_config["portfolio"]["max_weight"],
        "cash_buffer": params.get("cash_buffer") or base_config["portfolio"]["cash_buffer"],
        "target_annual_vol": params.get("target_annual_vol") or base_config["portfolio"]["target_annual_vol"],
    }


def _summary_cards(snapshot) -> list[dict]:
    return [
        {
            "label": "Total return",
            "value": f"{snapshot.metrics.total_return:.2%}",
        },
        {
            "label": "Annual return",
            "value": f"{snapshot.metrics.annual_return:.2%}",
        },
        {
            "label": "Volatility",
            "value": f"{snapshot.metrics.volatility_annual:.2%}",
        },
        {
            "label": "Sharpe",
            "value": f"{snapshot.metrics.sharpe_ratio:.2f}",
        },
        {
            "label": "Max drawdown",
            "value": f"{snapshot.metrics.max_drawdown:.2%}",
        },
        {
            "label": "Days traded",
            "value": f"{len(snapshot.result.returns):,}",
        },
    ]


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    base_config = load_config([])
    form_values = _query_values(request, base_config)
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
        "base_config": base_config,
        "form_values": form_values,
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
