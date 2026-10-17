"""Fill-loop tests: cost accounting and the zero-cost identity with buy&hold."""

import numpy as np
import pandas as pd
import pytest

from backtester import data as datalayer
from backtester.engine import (
    BacktestConfig,
    Commission,
    buy_and_hold,
    run_backtest,
)


def _bars(n=100, seed=3):
    return datalayer.synthetic_bars(n=n, seed=seed)


def test_zero_cost_buy_and_hold_identity():
    """With slippage = 0 and commission = 0, an always-long strategy's equity
    must equal the price series normalized to the first open, exactly."""
    bars = _bars()
    signal = pd.Series(np.ones(len(bars), dtype=int))
    config = BacktestConfig(
        initial_cash=1_000_000.0, slippage_bps=0.0, commission=Commission("flat", 0.0)
    )
    res = run_backtest(bars, signal, config)
    expected = 1_000_000.0 * bars["close"].to_numpy() / bars["open"].iloc[0]
    assert np.allclose(res.equity.to_numpy(), expected, rtol=1e-12)
    assert len(res.trades) == 1
    assert res.total_commission == 0.0


def test_flat_commission_accounting():
    """Total commission paid must equal fee x number of trades."""
    bars = _bars(200, seed=5)
    # alternating target forces multiple round trips
    signal = pd.Series((np.arange(len(bars)) // 10) % 2)
    config = BacktestConfig(
        initial_cash=1_000_000.0, slippage_bps=0.0, commission=Commission("flat", 20.0)
    )
    res = run_backtest(bars, signal, config)
    assert res.total_commission == pytest.approx(20.0 * len(res.trades))
    assert len(res.trades) > 2


def test_bps_commission_accounting():
    bars = _bars(200, seed=5)
    signal = pd.Series((np.arange(len(bars)) // 10) % 2)
    config = BacktestConfig(
        initial_cash=1_000_000.0, slippage_bps=0.0, commission=Commission("bps", 5.0)
    )
    res = run_backtest(bars, signal, config)
    expected = sum(t.value * 5.0 / 1e4 for t in res.trades)
    assert res.total_commission == pytest.approx(expected, rel=1e-12)


def test_slippage_direction():
    bars = _bars(60, seed=9)
    signal = pd.Series([1] * 30 + [0] * 30)
    config = BacktestConfig(
        initial_cash=1_000_000.0, slippage_bps=10.0, commission=Commission("flat", 0.0)
    )
    res = run_backtest(bars, signal, config)
    buy = next(t for t in res.trades if t.side == "buy")
    sell = next(t for t in res.trades if t.side == "sell")
    buy_bar_open = bars.loc[bars["timestamp"] == buy.timestamp, "open"].iloc[0]
    sell_bar_open = bars.loc[bars["timestamp"] == sell.timestamp, "open"].iloc[0]
    assert buy.price == pytest.approx(buy_bar_open * 1.001)
    assert sell.price == pytest.approx(sell_bar_open * 0.999)


def test_costs_reduce_returns_vs_frictionless():
    bars = _bars(300, seed=21)
    sig = pd.Series((np.arange(len(bars)) // 15) % 2)
    free = run_backtest(
        bars, sig, BacktestConfig(slippage_bps=0.0, commission=Commission("flat", 0.0))
    )
    costly = run_backtest(
        bars, sig, BacktestConfig(slippage_bps=10.0, commission=Commission("flat", 20.0))
    )
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_buy_and_hold_reference_shape():
    bars = _bars(50)
    bh = buy_and_hold(bars, 10_000.0)
    assert len(bh) == 50
    assert bh.iloc[-1] > 0
