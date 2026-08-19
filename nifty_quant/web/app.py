from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from nifty_quant.application.backtest_runner import build_backtest_snapshot, load_config
from nifty_quant.application.job_manager import JobManager, JobStatus
from nifty_quant.domain.strategies.registry import available_strategies


BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="Nifty-Quant Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

job_manager = JobManager.get_instance()

# Strategy-specific query params
_STRATEGY_PARAMS: dict[str, list[tuple[str, str]]] = {
    "momentum_12_1": [
        ("strategy.lookback_days",    "lookback_days"),
        ("strategy.skip_recent_days", "skip_recent_days"),
    ],
    "low_vol": [
        ("strategy.lookback_days",    "lookback_days"),
    ],
    "arima": [
        ("strategy.arima_p",   "arima_p"),
        ("strategy.fit_window","fit_window"),
        ("strategy.method",    "arima_method"),
    ],
}


def _dict_to_overrides(params: Dict[str, Any]) -> List[str]:
    overrides: List[str] = []
    strategy = str(params.get("strategy", "momentum_12_1"))
    overrides.append(f"strategy={strategy}")

    for key, qp in [
        ("backtest.start_date",      "start_date"),
        ("backtest.end_date",        "end_date"),
        ("backtest.initial_capital", "initial_capital"),
    ]:
        v = params.get(qp)
        if v not in (None, ""):
            overrides.append(f"{key}={v}")

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

    for key, qp in _STRATEGY_PARAMS.get(strategy, []):
        v = params.get(qp)
        if v not in (None, ""):
            overrides.append(f"{key}={v}")

    return overrides


def _query_overrides(request: Request) -> list[str]:
    return _dict_to_overrides(dict(request.query_params))


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


@app.post("/api/backtest/run")
async def run_backtest_api(payload: Optional[Dict[str, Any]] = Body(None), request: Request = None) -> JSONResponse:
    params: Dict[str, Any] = {}
    if payload:
        params.update(payload)
    if request and request.query_params:
        params.update(dict(request.query_params))

    overrides = _dict_to_overrides(params)
    job = job_manager.submit_job(overrides)
    return JSONResponse(job_manager.to_dict(job))


@app.get("/api/backtest/jobs/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job_manager.to_dict(job))


@app.websocket("/ws/backtest/{job_id}")
async def websocket_backtest_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    job = job_manager.get_job(job_id)
    if not job:
        await websocket.send_json({"error": "Job not found", "status": "FAILED"})
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    def listener(data: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, data)

    job_manager.add_listener(job_id, listener)
    await websocket.send_json(job_manager.to_dict(job))

    try:
        while True:
            data = await queue.get()
            await websocket.send_json(data)
            if data.get("status") in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.remove_listener(job_id, listener)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    base_cfg = load_config([])
    form_values = _query_values(request, base_cfg)

    context = {
        "request": request,
        "base_config": base_cfg,
        "form_values": form_values,
        "strategies": available_strategies(),
        "snapshot": None,
        "summary_cards": [],
        "error_message": None,
        "equity_end": None,
        "chart_path": "",
        "chart_min": 0.0,
        "chart_max": 0.0,
        "holdings": [],
    }
    return templates.TemplateResponse(request, "dashboard.html", context)

