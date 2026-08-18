"""Micro-benchmark comparing ONLY model fitting time: Batched OLS vs statsmodels ARIMA MLE."""

from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

from nifty_quant.domain.strategies.arima import _batch_ar_ols, _mle_forecast_one


def main():
    print("=" * 65)
    print(" MICRO-BENCHMARK: AR(2) MODEL FITTING TIME (50 STOCKS, 60-DAY WINDOW)")
    print("=" * 65)

    # 1. Create synthetic log-returns matrix for 50 stocks over 60 trading days
    np.random.seed(42)
    T, N =  60, 50
    log_returns = np.random.normal(loc=0.0005, scale=0.015, size=(T, N))
    p = 2
    order = (2, 0, 0)

    print(f"Data shape: {T} trading days x {N} stocks (total {N} AR({p}) regressions per step)")
    print("-" * 65)

    # 2. Benchmark Batched Vectorized OLS (einsum)
    # Warmup
    _batch_ar_ols(log_returns, p)

    n_runs_ols = 1000
    t0 = time.perf_counter()
    for _ in range(n_runs_ols):
        _batch_ar_ols(log_returns, p)
    t1 = time.perf_counter()

    avg_ols_step_sec = (t1 - t0) / n_runs_ols
    avg_ols_step_ms = avg_ols_step_sec * 1000
    avg_ols_per_stock_ms = avg_ols_step_ms / N

    print(f"\n1. Batched OLS (numpy einsum SIMD vectorization):")
    print(f"   Total time for {n_runs_ols:,} steps : {(t1 - t0):.4f} seconds")
    print(f"   Avg time per step (50 stocks)  : {avg_ols_step_ms:.3f} ms  ({avg_ols_step_sec:.6f} s)")
    print(f"   Avg time per stock            : {avg_ols_per_stock_ms:.4f} ms")

    # 3. Benchmark Parallel statsmodels MLE (ThreadPool max_workers=4)
    def _mle_fit_parallel():
        def _fit(col_idx):
            return _mle_forecast_one(log_returns[:, col_idx], order)
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(_fit, range(N)))

    # Warmup
    _mle_fit_parallel()

    n_runs_mle_parallel = 10
    t0 = time.perf_counter()
    for _ in range(n_runs_mle_parallel):
        _mle_fit_parallel()
    t1 = time.perf_counter()

    avg_mle_p_step_sec = (t1 - t0) / n_runs_mle_parallel
    avg_mle_p_step_ms = avg_mle_p_step_sec * 1000
    avg_mle_p_per_stock_ms = avg_mle_p_step_ms / N

    print(f"\n2. Parallel statsmodels MLE (4 Thread Pool Workers):")
    print(f"   Total time for {n_runs_mle_parallel} steps      : {(t1 - t0):.4f} seconds")
    print(f"   Avg time per step (50 stocks)  : {avg_mle_p_step_ms:.2f} ms  ({avg_mle_p_step_sec:.4f} s)")
    print(f"   Avg time per stock            : {avg_mle_p_per_stock_ms:.2f} ms")

    # 4. Benchmark Sequential statsmodels MLE (1 Thread / sequential)
    def _mle_fit_sequential():
        return [_mle_forecast_one(log_returns[:, i], order) for i in range(N)]

    n_runs_mle_seq = 3
    t0 = time.perf_counter()
    for _ in range(n_runs_mle_seq):
        _mle_fit_sequential()
    t1 = time.perf_counter()

    avg_mle_s_step_sec = (t1 - t0) / n_runs_mle_seq
    avg_mle_s_step_ms = avg_mle_s_step_sec * 1000

    print(f"\n3. Sequential statsmodels MLE (Single Thread):")
    print(f"   Total time for {n_runs_mle_seq} steps       : {(t1 - t0):.4f} seconds")
    print(f"   Avg time per step (50 stocks)  : {avg_mle_s_step_ms:.2f} ms  ({avg_mle_s_step_sec:.4f} s)")

    # 5. Speedup Ratios
    speedup_vs_parallel = avg_mle_p_step_sec / avg_ols_step_sec
    speedup_vs_sequential = avg_mle_s_step_sec / avg_ols_step_sec

    print("\n" + "=" * 65)
    print(f" SPEEDUP RESULTS:")
    print(f"   Batched OLS vs. 4-Thread MLE : {speedup_vs_parallel:>8.1f}x FASTER")
    print(f"   Batched OLS vs. 1-Thread MLE : {speedup_vs_sequential:>8.1f}x FASTER")
    print("=" * 65)


if __name__ == "__main__":
    main()
