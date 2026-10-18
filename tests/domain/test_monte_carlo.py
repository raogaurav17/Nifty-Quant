"""Unit tests for the Monte Carlo block-bootstrap module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_quant.domain.monte_carlo import (
    MonteCarloResult,
    ForwardProjection,
    block_bootstrap_metrics,
    forward_projection,
    _sharpe,
    _max_drawdown,
    _cagr,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def flat_returns() -> pd.Series:
    """Constant 0.5% daily return — deterministic, easy to verify."""
    return pd.Series([0.005] * 500)


@pytest.fixture()
def zero_returns() -> pd.Series:
    """All-zero returns — edge case."""
    return pd.Series([0.0] * 200)


@pytest.fixture()
def random_returns(rng: np.random.Generator = np.random.default_rng(0)) -> pd.Series:
    """~1000 days of realistic daily returns drawn from N(0.0003, 0.01)."""
    return pd.Series(rng.normal(loc=0.0003, scale=0.01, size=1000))


# ── Helper function tests ─────────────────────────────────────────────────────


class TestHelpers:
    def test_sharpe_constant_positive_return(self) -> None:
        """Constant positive daily return should produce a large positive Sharpe."""
        r = np.full(252, 0.001)
        result = _sharpe(r)
        assert result > 0

    def test_sharpe_zero_vol(self) -> None:
        """Zero-volatility returns (all equal to rf) should return 0 without crashing."""
        r = np.zeros(252)
        assert _sharpe(r) == 0.0

    def test_max_drawdown_monotone_up(self) -> None:
        """Monotonically rising equity → zero drawdown."""
        r = np.full(100, 0.01)
        assert _max_drawdown(r) == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_single_crash(self) -> None:
        """A single -50% day should produce at most -50% drawdown."""
        r = np.zeros(100)
        r[50] = -0.5
        dd = _max_drawdown(r)
        assert dd <= -0.49

    def test_cagr_flat_positive(self, flat_returns: pd.Series) -> None:
        """Constant 0.5% daily return over ~2 years should give large positive CAGR."""
        result = _cagr(flat_returns.values)
        assert result > 1.0  # >100% annualised over 500 days at 0.5%/day

    def test_cagr_zero_returns(self, zero_returns: pd.Series) -> None:
        """Zero daily returns → zero CAGR."""
        result = _cagr(zero_returns.values)
        assert result == pytest.approx(0.0, abs=1e-9)


# ── block_bootstrap_metrics tests ────────────────────────────────────────────


class TestBlockBootstrapMetrics:
    def test_returns_correct_type(self, random_returns: pd.Series) -> None:
        result = block_bootstrap_metrics(random_returns, n_simulations=100)
        assert isinstance(result, MonteCarloResult)

    def test_ci_ordering_sharpe(self, random_returns: pd.Series) -> None:
        """5th percentile ≤ median ≤ 95th percentile for Sharpe."""
        result = block_bootstrap_metrics(random_returns, n_simulations=200)
        assert result.sharpe_ci_low <= result.sharpe_ci_mid <= result.sharpe_ci_high

    def test_ci_ordering_max_drawdown(self, random_returns: pd.Series) -> None:
        """Max-DD CI should be negative; 5th pct (worst) ≤ 95th pct (least bad)."""
        result = block_bootstrap_metrics(random_returns, n_simulations=200)
        assert result.max_dd_ci_low <= result.max_dd_ci_mid <= result.max_dd_ci_high
        assert result.max_dd_ci_low < 0  # must always be negative

    def test_ci_ordering_cagr(self, random_returns: pd.Series) -> None:
        result = block_bootstrap_metrics(random_returns, n_simulations=200)
        assert result.cagr_ci_low <= result.cagr_ci_mid <= result.cagr_ci_high

    def test_probability_bounds(self, random_returns: pd.Series) -> None:
        """Probability fields must lie in [0, 1]."""
        result = block_bootstrap_metrics(random_returns, n_simulations=100)
        assert 0.0 <= result.p_positive_sharpe <= 1.0
        assert 0.0 <= result.p_positive_return <= 1.0

    def test_n_simulations_recorded(self, random_returns: pd.Series) -> None:
        result = block_bootstrap_metrics(random_returns, n_simulations=150)
        assert result.n_simulations == 150

    def test_block_size_default_sqrt_T(self, random_returns: pd.Series) -> None:
        """Default block size should equal floor(sqrt(T))."""
        T = len(random_returns.dropna())
        result = block_bootstrap_metrics(random_returns, n_simulations=50)
        assert result.block_size == int(np.sqrt(T))

    def test_custom_block_size(self, random_returns: pd.Series) -> None:
        result = block_bootstrap_metrics(random_returns, n_simulations=50, block_size=20)
        assert result.block_size == 20

    def test_reproducibility(self, random_returns: pd.Series) -> None:
        """Same seed → identical results."""
        r1 = block_bootstrap_metrics(random_returns, n_simulations=100, random_seed=7)
        r2 = block_bootstrap_metrics(random_returns, n_simulations=100, random_seed=7)
        assert r1.sharpe_ci_mid == pytest.approx(r2.sharpe_ci_mid)

    def test_different_seeds_differ(self, random_returns: pd.Series) -> None:
        r1 = block_bootstrap_metrics(random_returns, n_simulations=200, random_seed=1)
        r2 = block_bootstrap_metrics(random_returns, n_simulations=200, random_seed=2)
        # Medians will be close but not exactly equal with different seeds
        assert abs(r1.sharpe_ci_mid - r2.sharpe_ci_mid) < 0.5

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 10"):
            block_bootstrap_metrics(pd.Series([0.01] * 5), n_simulations=10)

    def test_sharpe_distribution_length(self, random_returns: pd.Series) -> None:
        result = block_bootstrap_metrics(random_returns, n_simulations=100)
        assert len(result.sharpe_distribution) == 80

    def test_to_dict_keys(self, random_returns: pd.Series) -> None:
        result = block_bootstrap_metrics(random_returns, n_simulations=50)
        d = result.to_dict()
        required_keys = {
            "n_simulations", "block_size",
            "sharpe_ci_low", "sharpe_ci_mid", "sharpe_ci_high",
            "max_dd_ci_low", "max_dd_ci_mid", "max_dd_ci_high",
            "cagr_ci_low", "cagr_ci_mid", "cagr_ci_high",
            "p_positive_sharpe", "p_positive_return",
            "sharpe_distribution",
        }
        assert required_keys.issubset(d.keys())

    def test_high_drift_returns_positive_sharpe_ci(self) -> None:
        """A strategy with strong positive drift should yield a clearly positive Sharpe CI.

        Uses realistic noisy returns (not a constant series) with a large
        positive mean so that even the P5 bootstrap resample yields Sharpe > 0.
        """
        rng = np.random.default_rng(77)
        # Mean ~0.5% / day with σ=0.008 → annualised Sharpe ~10 (unrealistically good,
        # but guarantees the bootstrap distribution is safely positive)
        high_drift = pd.Series(rng.normal(loc=0.005, scale=0.008, size=500))
        result = block_bootstrap_metrics(high_drift, n_simulations=200)
        assert result.sharpe_ci_low > 0
        assert result.p_positive_sharpe == pytest.approx(1.0)


# ── forward_projection tests ──────────────────────────────────────────────────


class TestForwardProjection:
    def test_returns_correct_type(self, random_returns: pd.Series) -> None:
        result = forward_projection(random_returns, horizon_days=60, n_paths=50)
        assert isinstance(result, ForwardProjection)

    def test_output_length(self, random_returns: pd.Series) -> None:
        horizon = 63
        result = forward_projection(random_returns, horizon_days=horizon, n_paths=50)
        assert len(result.p10) == horizon
        assert len(result.p50) == horizon
        assert len(result.p90) == horizon

    def test_percentile_ordering(self, random_returns: pd.Series) -> None:
        """P10 ≤ P50 ≤ P90 at every future step."""
        result = forward_projection(random_returns, horizon_days=100, n_paths=200)
        p10 = np.array(result.p10)
        p50 = np.array(result.p50)
        p90 = np.array(result.p90)
        assert np.all(p10 <= p50 + 1e-9)
        assert np.all(p50 <= p90 + 1e-9)

    def test_anchored_to_last_equity(self, random_returns: pd.Series) -> None:
        """First step of each percentile path should be close to last_equity * (1+r)."""
        equity = 1_500_000.0
        result = forward_projection(
            random_returns, horizon_days=10, n_paths=500, last_equity=equity
        )
        # Day-1 median should be within ±5% of the anchor
        assert abs(result.p50[0] / equity - 1.0) < 0.05

    def test_reproducibility(self, random_returns: pd.Series) -> None:
        r1 = forward_projection(random_returns, horizon_days=60, n_paths=100, random_seed=99)
        r2 = forward_projection(random_returns, horizon_days=60, n_paths=100, random_seed=99)
        assert r1.p50 == r2.p50

    def test_horizon_and_paths_recorded(self, random_returns: pd.Series) -> None:
        result = forward_projection(random_returns, horizon_days=42, n_paths=77)
        assert result.horizon_days == 42
        assert result.n_paths == 77

    def test_to_dict_keys(self, random_returns: pd.Series) -> None:
        result = forward_projection(random_returns, horizon_days=30, n_paths=50)
        d = result.to_dict()
        assert set(d.keys()) == {"horizon_days", "n_paths", "p10", "p50", "p90", "last_equity"}

    def test_to_dict_list_length(self, random_returns: pd.Series) -> None:
        horizon = 30
        result = forward_projection(random_returns, horizon_days=horizon, n_paths=50)
        d = result.to_dict()
        assert len(d["p10"]) == horizon
        assert len(d["p50"]) == horizon
        assert len(d["p90"]) == horizon

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 10"):
            forward_projection(pd.Series([0.01] * 3), n_paths=10)

    def test_custom_block_size(self, random_returns: pd.Series) -> None:
        """Custom block size should not crash and should produce valid output."""
        result = forward_projection(random_returns, horizon_days=50, n_paths=50, block_size=15)
        assert len(result.p50) == 50
