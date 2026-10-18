"""Asynchronous job manager for background backtest and Monte Carlo execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time
from typing import Any, Callable
import uuid

import pandas as pd

from nifty_quant.application.backtest_runner import BacktestSnapshot, build_backtest_snapshot
from nifty_quant.domain.monte_carlo import (
    MonteCarloResult,
    ForwardProjection,
    block_bootstrap_metrics,
    forward_projection,
)

logger = logging.getLogger(__name__)


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


@dataclass
class MCJob:
    """State and progress container for a Monte Carlo simulation task."""

    job_id: str
    created_at: float
    status: JobStatus
    progress: float
    status_message: str
    source_job_id: str          # the completed BacktestJob whose returns we bootstrap
    n_simulations: int
    horizon_days: int
    mc_result: MonteCarloResult | None = None
    forward: ForwardProjection | None = None
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

    def submit_mc_job(
        self,
        source_job_id: str,
        n_simulations: int = 2000,
        horizon_days: int = 252,
    ) -> MCJob:
        """Submit a Monte Carlo simulation job for a completed backtest.

        The MC job reads the daily returns from the source BacktestJob and runs
        block_bootstrap_metrics + forward_projection in a background thread,
        streaming progress via the same listener / WebSocket infrastructure.

        Args:
            source_job_id: ID of a COMPLETED BacktestJob whose returns to resample.
            n_simulations:  Number of bootstrap resamples for CI estimation.
            horizon_days:   Forward projection horizon in trading days.

        Returns:
            MCJob instance tracking the scheduled Monte Carlo task.

        Raises:
            ValueError: If the source job is not found or not yet completed.
        """
        with self._jobs_lock:
            source = self.jobs.get(source_job_id)
        if source is None or not isinstance(source, BacktestJob):
            raise ValueError(f"Source backtest job '{source_job_id}' not found.")
        if source.status != JobStatus.COMPLETED or source.snapshot is None:
            raise ValueError(
                f"Source job '{source_job_id}' is not completed (status={source.status}).")

        mc_job_id = str(uuid.uuid4())
        mc_job = MCJob(
            job_id=mc_job_id,
            created_at=time.time(),
            status=JobStatus.QUEUED,
            progress=0.0,
            status_message="Monte Carlo job queued...",
            source_job_id=source_job_id,
            n_simulations=n_simulations,
            horizon_days=horizon_days,
        )
        with self._jobs_lock:
            self.jobs[mc_job_id] = mc_job  # type: ignore[assignment]
            self._prune_old_jobs()

        self.executor.submit(self._execute_mc_job, mc_job_id)
        return mc_job

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
                logger.exception("Job listener failed for job %s", job.job_id)

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

    def _execute_mc_job(self, mc_job_id: str) -> None:
        """Worker thread entry point executing the Monte Carlo simulation."""
        with self._jobs_lock:
            mc_job = self.jobs.get(mc_job_id)
            source = self.jobs.get(mc_job.source_job_id) if mc_job else None  # type: ignore[union-attr]
        if mc_job is None or source is None or not isinstance(source, BacktestJob):
            return

        mc_job.status = JobStatus.RUNNING
        mc_job.progress = 0.05
        mc_job.status_message = "Extracting backtest returns for resampling..."
        self._notify_listeners(mc_job)

        try:
            returns: pd.Series = source.snapshot.result.returns  # type: ignore[union-attr]
            last_equity: float = float(source.snapshot.result.equity_curve.iloc[-1])  # type: ignore[union-attr]

            mc_job.progress = 0.15
            mc_job.status_message = (
                f"Running block-bootstrap CI ({mc_job.n_simulations:,} simulations)..."
            )
            self._notify_listeners(mc_job)

            mc_result = block_bootstrap_metrics(
                returns=returns,
                n_simulations=mc_job.n_simulations,
            )

            mc_job.progress = 0.65
            mc_job.status_message = (
                f"Generating forward projection ({mc_job.horizon_days}-day fan)..."
            )
            self._notify_listeners(mc_job)

            fwd = forward_projection(
                returns=returns,
                horizon_days=mc_job.horizon_days,
                n_paths=1000,
                last_equity=last_equity,
            )

            mc_job.mc_result = mc_result
            mc_job.forward = fwd
            mc_job.status = JobStatus.COMPLETED
            mc_job.progress = 1.0
            mc_job.status_message = "Monte Carlo simulation completed."
        except Exception as exc:
            mc_job.status = JobStatus.FAILED
            mc_job.progress = 1.0
            mc_job.error = str(exc)
            mc_job.status_message = f"Monte Carlo failed: {exc}"
            logger.exception("MC job %s failed", mc_job_id)
        finally:
            self._notify_listeners(mc_job)

    def _prune_old_jobs(self, max_keep: int = 50) -> None:
        """Maintain memory footprint by pruning oldest jobs when exceeding max_keep."""
        if len(self.jobs) > max_keep:
            sorted_keys = sorted(self.jobs.keys(), key=lambda k: self.jobs[k].created_at)
            for k in sorted_keys[: len(self.jobs) - max_keep]:
                del self.jobs[k]

    def to_dict(self, job: BacktestJob | MCJob) -> dict[str, Any]:  # type: ignore[override]
        """Serialize job state and results into a JSON-compatible dictionary."""
        data: dict[str, Any] = {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "status": job.status.value,
            "progress": round(job.progress, 3),
            "status_message": job.status_message,
            "error": job.error,
        }

        if isinstance(job, MCJob):
            data["job_type"] = "monte_carlo"
            if job.mc_result is not None:
                data["mc_result"] = job.mc_result.to_dict()
            if job.forward is not None:
                data["forward_projection"] = job.forward.to_dict()
            return data

        # --- BacktestJob serialisation ---
        data["job_type"] = "backtest"
        if job.snapshot is not None:
            snapshot = job.snapshot
            data["snapshot"] = {
                "strategy_name": snapshot.config.get("strategy", {}).get("name", "momentum_12_1"),
                "metrics": {
                    "total_return": float(snapshot.metrics.total_return),
                    "annual_return": float(snapshot.metrics.annual_return),
                    "volatility_annual": float(snapshot.metrics.volatility_annual),
                    "sharpe_ratio": float(snapshot.metrics.sharpe_ratio),
                    "sortino_ratio": float(snapshot.metrics.sortino_ratio),
                    "max_drawdown": float(snapshot.metrics.max_drawdown),
                    "calmar_ratio": (
                        float(snapshot.metrics.calmar_ratio)
                        if snapshot.metrics.calmar_ratio is not None
                        else None
                    ),
                    "days_traded": len(snapshot.result.returns),
                },
                "chart_path": snapshot.chart_path,
                "chart_points": snapshot.chart_points,
                "chart_min": snapshot.chart_min,
                "chart_max": snapshot.chart_max,
                "equity_end": (
                    float(snapshot.result.equity_curve.iloc[-1])
                    if not snapshot.result.equity_curve.empty
                    else 0.0
                ),
                "holdings": [
                    {
                        **holding,
                        "forecast": snapshot.forecasts.get(holding["symbol"]),
                    }
                    for holding in snapshot.holdings
                ],
                "recent_trades": snapshot.recent_trades,
                "start_date": str(snapshot.start_date),
                "end_date": str(snapshot.end_date) if snapshot.end_date else "present",
            }

        return data
