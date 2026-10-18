import time
import logging
from unittest.mock import patch

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient

from nifty_quant.application.job_manager import BacktestJob, JobManager, JobStatus
from nifty_quant.web.app import app


@pytest.fixture(autouse=True)
def mock_yahoo_prices(tmp_path):
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    fake_df1 = pd.DataFrame({"adj_close": [100.0 + i for i in range(100)], "volume": [1000] * 100}, index=dates)
    fake_df2 = pd.DataFrame({"adj_close": [200.0 + i * 2 for i in range(100)], "volume": [1000] * 100}, index=dates)

    data = {
        "RELIANCE.NS": fake_df1,
        "TCS.NS": fake_df2,
    }

    def fake_get_prices(self, symbols, start_date, end_date=None):
        return {s: data.get(s, fake_df1) for s in symbols}

    def fake_cached_init(self, upstream, cache_dir="data/cache", force_refresh=False):
        self.upstream = upstream
        self.cache_dir = tmp_path
        self.force_refresh = force_refresh
        from nifty_quant.infrastructure.data.date_intervals import IntervalRegistry
        from nifty_quant.infrastructure.data.local_price_store import LocalPriceStore
        self.registry = IntervalRegistry(self.cache_dir / "intervals.json")
        self.store = LocalPriceStore(self.cache_dir)

    with (
        patch("nifty_quant.application.backtest_runner.YahooPriceRepository.get_prices", new=fake_get_prices),
        patch("nifty_quant.infrastructure.data.yahoo_price_repository.YahooPriceRepository.get_prices", new=fake_get_prices),
        patch("nifty_quant.application.backtest_runner.CachedPriceRepository.__init__", new=fake_cached_init),
        patch("nifty_quant.infrastructure.data.cached_price_repository.CachedPriceRepository.__init__", new=fake_cached_init),
    ):
        yield


def test_job_manager_lifecycle():
    manager = JobManager.get_instance()
    overrides = ["strategy=momentum_12_1", "backtest.start_date=2024-01-01", "backtest.end_date=2024-03-01"]

    job = manager.submit_job(overrides)
    assert job.job_id is not None
    assert job.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED)

    # Wait for completion
    start_wait = time.time()
    while job.status in (JobStatus.QUEUED, JobStatus.RUNNING) and (time.time() - start_wait) < 5:
        time.sleep(0.05)

    assert job.status == JobStatus.COMPLETED
    assert job.progress == 1.0
    assert job.snapshot is not None


def test_listener_failures_are_logged(caplog):
    manager = JobManager(max_workers=1)
    job = BacktestJob(
        job_id="listener-test",
        created_at=time.time(),
        status=JobStatus.QUEUED,
        progress=0.0,
        status_message="queued",
        overrides=[],
    )
    manager.jobs[job.job_id] = job

    def failing_listener(data):
        del data
        raise RuntimeError("listener failed")

    manager.add_listener(job.job_id, failing_listener)

    with caplog.at_level(logging.ERROR, logger="nifty_quant.application.job_manager"):
        manager._notify_listeners(job)

    assert "Job listener failed" in caplog.text
    manager.executor.shutdown(wait=True)


@pytest.mark.anyio
async def test_web_api_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        payload = {
            "strategy": "momentum_12_1",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "top_k": "5",
            "vol_lookback_days": "30",
        }

        response = await client.post("/api/backtest/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ("QUEUED", "RUNNING", "COMPLETED")

        job_id = data["job_id"]

        # Test GET /api/backtest/jobs/{job_id}
        status_response = await client.get(f"/api/backtest/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == job_id


@pytest.mark.anyio
async def test_job_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/api/backtest/jobs/non-existent-id")
        assert response.status_code == 404


def test_websocket_returns_already_completed_job():
    manager = JobManager.get_instance()
    job = BacktestJob(
        job_id="already-completed",
        created_at=time.time(),
        status=JobStatus.COMPLETED,
        progress=1.0,
        status_message="done",
        overrides=[],
    )
    manager.jobs[job.job_id] = job

    with TestClient(app) as client:
        with client.websocket_connect("/ws/backtest/already-completed") as websocket:
            data = websocket.receive_json()

    assert data["status"] == JobStatus.COMPLETED.value
    assert data["progress"] == 1.0
    manager.jobs.pop(job.job_id, None)


def test_mc_job_lifecycle():
    manager = JobManager.get_instance()
    overrides = ["strategy=momentum_12_1", "backtest.start_date=2024-01-01", "backtest.end_date=2024-03-01"]

    bt_job = manager.submit_job(overrides)
    start_wait = time.time()
    while bt_job.status in (JobStatus.QUEUED, JobStatus.RUNNING) and (time.time() - start_wait) < 5:
        time.sleep(0.05)
    assert bt_job.status == JobStatus.COMPLETED

    mc_job = manager.submit_mc_job(source_job_id=bt_job.job_id, n_simulations=50, horizon_days=30)
    assert mc_job.job_id is not None
    assert mc_job.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED)

    start_wait = time.time()
    while mc_job.status in (JobStatus.QUEUED, JobStatus.RUNNING) and (time.time() - start_wait) < 5:
        time.sleep(0.05)

    assert mc_job.status == JobStatus.COMPLETED
    assert mc_job.mc_result is not None
    assert mc_job.forward is not None
    assert mc_job.mc_result.n_simulations == 50
    assert len(mc_job.forward.p50) == 30


@pytest.mark.anyio
async def test_mc_web_api_endpoints():
    manager = JobManager.get_instance()
    overrides = ["strategy=momentum_12_1", "backtest.start_date=2024-01-01", "backtest.end_date=2024-03-01"]
    bt_job = manager.submit_job(overrides)

    start_wait = time.time()
    while bt_job.status in (JobStatus.QUEUED, JobStatus.RUNNING) and (time.time() - start_wait) < 5:
        time.sleep(0.05)
    assert bt_job.status == JobStatus.COMPLETED

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # Invalid source job
        bad_resp = await client.post("/api/mc/run", json={"source_job_id": "nonexistent"})
        assert bad_resp.status_code == 400

        # Valid source job
        payload = {"source_job_id": bt_job.job_id, "n_simulations": 50, "horizon_days": 30}
        resp = await client.post("/api/mc/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        mc_job_id = data["job_id"]

        status_resp = await client.get(f"/api/mc/{mc_job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["job_id"] == mc_job_id

