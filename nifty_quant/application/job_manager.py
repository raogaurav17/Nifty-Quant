"""Asynchronous job manager for background backtest execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Any, Callable
import uuid

from nifty_quant.application.backtest_runner import BacktestSnapshot, build_backtest_snapshot


class JobStatus(str, Enum):
    """Execution status for asynchronous backtest jobs."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class BacktestJob:
    """State and progress container for a single backtest execution task."""

    job_id: str
    created_at: float
    status: JobStatus
    progress: float
    status_message: str
    overrides: list[str]
    snapshot: BacktestSnapshot | None = None
    error: str | None = None
    listeners: set[Callable[[dict[str, Any]], None]] = field(default_factory=set, repr=False)


class JobManager:
    """Thread-safe manager for scheduling and tracking asynchronous backtest tasks."""

    _instance: JobManager | None = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nifty_quant_worker")
        self.jobs: dict[str, BacktestJob] = {}
        self._jobs_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> JobManager:
        """Singleton accessor for the global JobManager instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = JobManager()
            return cls._instance

    def submit_job(self, overrides: list[str]) -> BacktestJob:
        """Create and submit a backtest job to the worker pool.

        Args:
            overrides: List of Hydra configuration override strings.

        Returns:
            BacktestJob instance tracking the scheduled task.
        """
        job_id = str(uuid.uuid4())
        job = BacktestJob(
            job_id=job_id,
            created_at=time.time(),
            status=JobStatus.QUEUED,
            progress=0.0,
            status_message="Job queued in worker pool...",
            overrides=overrides,
        )
        with self._jobs_lock:
            self.jobs[job_id] = job
            self._prune_old_jobs()

        self.executor.submit(self._execute_job, job_id)
        return job

    def get_job(self, job_id: str) -> BacktestJob | None:
        """Retrieve a job by its unique identifier."""
        with self._jobs_lock:
            return self.jobs.get(job_id)

    def add_listener(self, job_id: str, callback: Callable[[dict[str, Any]], None]) -> bool:
        """Attach a notification callback listener to a specific job."""
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            job.listeners.add(callback)
            return True

    def remove_listener(self, job_id: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Remove a notification callback listener from a job."""
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if job:
                job.listeners.discard(callback)

    def _notify_listeners(self, job: BacktestJob) -> None:
        """Broadcast updated job dictionary payload to all registered listeners."""
        payload = self.to_dict(job)
        with self._jobs_lock:
            listeners_copy = list(job.listeners)
        for listener in listeners_copy:
            try:
                listener(payload)
            except Exception:
                pass

    def _execute_job(self, job_id: str) -> None:
        """Worker thread entry point executing the backtest pipeline."""
        with self._jobs_lock:
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
        """Maintain memory footprint by pruning oldest jobs when exceeding max_keep."""
        if len(self.jobs) > max_keep:
            sorted_keys = sorted(self.jobs.keys(), key=lambda k: self.jobs[k].created_at)
            for k in sorted_keys[: len(self.jobs) - max_keep]:
                del self.jobs[k]

    def to_dict(self, job: BacktestJob) -> dict[str, Any]:
        """Serialize job state and snapshot into a JSON-compatible dictionary."""
        data: dict[str, Any] = {
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
                    "calmar_ratio": f"{snapshot.metrics.calmar_ratio:.2f}" if snapshot.metrics.calmar_ratio is not None else None,
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

