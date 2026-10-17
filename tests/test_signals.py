"""Signal-layer tests: EMA against a hand-rolled reference, and the
look-ahead truncation test (the most valuable test in the repo)."""

import numpy as np
import pandas as pd
import pytest

from backtester import data as datalayer
from backtester.engine import BacktestConfig, Commission, run_backtest
from backtester.signals import ema, ema_crossover, trading_signal


def test_ema_matches_recursive_reference():
    rng = np.random.default_rng(7)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    span = 20
    alpha = 2 / (span + 1)
    ref = np.empty(len(close))
    ref[0] = close.iloc[0]
    for i in range(1, len(close)):
        ref[i] = alpha * close.iloc[i] + (1 - alpha) * ref[i - 1]
    assert np.allclose(ema(close, span).to_numpy(), ref, atol=1e-10)


def test_ema_crossover_binary_and_requires_fast_lt_slow():
    close = pd.Series(np.linspace(1, 2, 50))
    sig = ema_crossover(close, 5, 20)
    assert set(sig.unique()) <= {0, 1}
    with pytest.raises(ValueError):
        ema_crossover(close, 20, 5)


def test_trading_signal_is_shifted():
    close = pd.Series(np.sin(np.linspace(0, 6, 200)) + 2)
    raw = ema_crossover(close, 5, 20)
    sig = trading_signal(close, 5, 20)
    assert sig.iloc[0] == 0
    assert np.array_equal(sig.iloc[1:].to_numpy(), raw.iloc[:-1].to_numpy())


def test_no_lookahead_truncation():
    """Signals and results over a shared window must not change when future
    data is removed. If cutting off the future changes the past, future data
    is leaking into past decisions: look-ahead bias."""
    bars = datalayer.synthetic_bars(n=400, seed=11)
    k = 250
    full_sig = trading_signal(bars["close"], 12, 48)
    trunc_bars = bars.iloc[:k].reset_index(drop=True)
    trunc_sig = trading_signal(trunc_bars["close"], 12, 48)
    assert np.array_equal(full_sig.iloc[:k].to_numpy(), trunc_sig.to_numpy())

    config = BacktestConfig(slippage_bps=5.0, commission=Commission("flat", 20.0))
    full_res = run_backtest(bars, full_sig, config)
    trunc_res = run_backtest(trunc_bars, trunc_sig, config)
    assert np.allclose(
        full_res.equity.iloc[:k].to_numpy(), trunc_res.equity.to_numpy(), atol=1e-9
    )
