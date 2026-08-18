from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from nifty_quant.application.backtest_runner import BacktestSnapshot, build_backtest_snapshot


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class BacktestJob:
    job_id: str
    created_at: float
    status: JobStatus
    progress: float
    status_message: str
    overrides: List[str]
    snapshot: Optional[BacktestSnapshot] = None
    error: Optional[str] = None
    listeners: Set[Callable[[Dict[str, Any]], None]] = field(default_factory=set, repr=False)


class JobManager:
    _instance: Optional[JobManager] = None

    def __init__(self, max_workers: int = 4) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nifty_quant_worker")
        self.jobs: Dict[str, BacktestJob] = {}

    @classmethod
    def get_instance(cls) -> JobManager:
        if cls._instance is None:
            cls._instance = JobManager()
        return cls._instance

    def submit_job(self, overrides: List[str]) -> BacktestJob:
        job_id = str(uuid.uuid4())
        job = BacktestJob(
            job_id=job_id,
            created_at=time.time(),
            status=JobStatus.QUEUED,
            progress=0.0,
            status_message="Job queued in worker pool...",
            overrides=overrides,
        )
        self.jobs[job_id] = job
        self._prune_old_jobs()

        self.executor.submit(self._execute_job, job_id)
        return job

    def get_job(self, job_id: str) -> Optional[BacktestJob]:
        return self.jobs.get(job_id)

    def add_listener(self, job_id: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        job.listeners.add(callback)
        return True

    def remove_listener(self, job_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        job = self.jobs.get(job_id)
        if job:
            job.listeners.discard(callback)

    def _notify_listeners(self, job: BacktestJob) -> None:
        payload = self.to_dict(job)
        listeners_copy = list(job.listeners)
        for listener in listeners_copy:
            try:
                listener(payload)
            except Exception:
                pass

    def _execute_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return

        job.status = JobStatus.RUNNING
        job.progress = 0.02
        job.status_message = "Starting background backtest runner..."
        self._notify_listeners(job)

        def progress_cb(pct: float, message: str) -> None:
            job.progress = min(max(pct, 0.0), 0.99)
            job.status_message = message
            self._notify_listeners(job)

        try:
            snapshot = build_backtest_snapshot(
                cfg=job.overrides,
                progress_callback=progress_cb,
            )
            job.snapshot = snapshot
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.status_message = "Backtest execution completed successfully."
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.progress = 1.0
            job.error = str(exc)
            job.status_message = f"Backtest failed: {exc}"
        finally:
            self._notify_listeners(job)

    def _prune_old_jobs(self, max_keep: int = 50) -> None:
        if len(self.jobs) > max_keep:
            sorted_keys = sorted(self.jobs.keys(), key=lambda k: self.jobs[k].created_at)
            for k in sorted_keys[: len(self.jobs) - max_keep]:
                del self.jobs[k]

    def to_dict(self, job: BacktestJob) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "status": job.status.value,
            "progress": round(job.progress, 3),
            "status_message": job.status_message,
            "error": job.error,
        }

        if job.snapshot is not None:
            snapshot = job.snapshot
            data["snapshot"] = {
                "strategy_name": snapshot.config.get("strategy", {}).get("name", "momentum_12_1"),
                "metrics": {
                    "total_return": f"{snapshot.metrics.total_return:.2%}",
                    "annual_return": f"{snapshot.metrics.annual_return:.2%}",
                    "volatility_annual": f"{snapshot.metrics.volatility_annual:.2%}",
                    "sharpe_ratio": f"{snapshot.metrics.sharpe_ratio:.2f}",
                    "sortino_ratio": f"{snapshot.metrics.sortino_ratio:.2f}",
                    "max_drawdown": f"{snapshot.metrics.max_drawdown:.2%}",
                    "calmar_ratio": f"{snapshot.metrics.calmar_ratio:.2f}",
                    "days_traded": len(snapshot.result.returns),
                },
                "chart_path": snapshot.chart_path,
                "chart_min": snapshot.chart_min,
                "chart_max": snapshot.chart_max,
                "equity_end": snapshot.result.equity_curve.iloc[-1] if not snapshot.result.equity_curve.empty else 0.0,
                "holdings": snapshot.holdings,
                "recent_trades": snapshot.recent_trades,
                "start_date": str(snapshot.start_date),
                "end_date": str(snapshot.end_date) if snapshot.end_date else "present",
            }

        return data
