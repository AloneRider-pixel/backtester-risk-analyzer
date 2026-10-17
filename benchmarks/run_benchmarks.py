"""Benchmarks. Every performance number in the README comes from this script.

Method: time.perf_counter() around the section, 5 runs, median reported.
Run: python benchmarks/run_benchmarks.py
"""

import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from backtester import data as datalayer
from backtester.engine import BacktestConfig, Commission, run_backtest
from backtester.signals import trading_signal

N_BARS = 500_000
RUNS = 5


def bench(fn, runs=RUNS):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    print(f"bars: {N_BARS}, runs: {RUNS}, reporting median")
    bars = datalayer.synthetic_bars(n=N_BARS, seed=99)
    close = bars["close"]

    t_signal = bench(lambda: trading_signal(close, 12, 48))
    print(f"signal generation (EMA crossover, vectorized): {t_signal*1000:.1f} ms")

    sig = trading_signal(close, 12, 48)
    config = BacktestConfig(slippage_bps=5.0, commission=Commission("flat", 20.0))
    t_fill = bench(lambda: run_backtest(bars, sig, config))
    print(f"full backtest incl. fill loop (pure Python):    {t_fill*1000:.1f} ms")

    with tempfile.TemporaryDirectory() as tmp:
        datalayer.save_parquet(bars, tmp)
        t_duck = bench(lambda: datalayer.query_bars("SYNTH", tmp, start="2020-01-01", end="2020-12-31"))
        print(f"DuckDB range query over {N_BARS} Parquet rows:  {t_duck*1000:.1f} ms")


if __name__ == "__main__":
    main()
