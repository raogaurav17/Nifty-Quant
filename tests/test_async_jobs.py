import time
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from nifty_quant.application.job_manager import JobManager, JobStatus
from nifty_quant.web.app import app


@pytest.fixture(autouse=True)
def mock_yahoo_prices():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    fake_df1 = pd.DataFrame({"adj_close": [100.0 + i for i in range(100)], "volume": [1000] * 100}, index=dates)
    fake_df2 = pd.DataFrame({"adj_close": [200.0 + i * 2 for i in range(100)], "volume": [1000] * 100}, index=dates)

    data = {
        "RELIANCE.NS": fake_df1,
        "TCS.NS": fake_df2,
    }

    def fake_get_prices(self, symbols, start_date, end_date=None):
        return {s: data.get(s, fake_df1) for s in symbols}

    with patch("nifty_quant.infrastructure.data.yahoo_price_repository.YahooPriceRepository.get_prices", new=fake_get_prices):
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



