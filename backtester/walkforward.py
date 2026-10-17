"""Walk-forward evaluation: rolling train/validate windows.

Parameters are chosen on each train window, then evaluated once on the
following unseen validate window. The concatenated validate equity curve is
the honest headline result; an in-sample optimized curve is noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import BacktestConfig, run_backtest
from .metrics import period_returns, sharpe_ratio
from .signals import trading_signal


@dataclass(frozen=True)
class Window:
    train_start: int
    train_end: int  # exclusive
    val_start: int
    val_end: int  # exclusive


def make_windows(n: int, train: int, validate: int, step: int | None = None) -> list[Window]:
    step = step or validate
    windows = []
    start = 0
    while start + train + validate <= n:
        windows.append(Window(start, start + train, start + train, start + train + validate))
        start += step
    return windows


def choose_params(
    bars: pd.DataFrame,
    fast_spans: list[int],
    slow_spans: list[int],
    config: BacktestConfig | None = None,
) -> tuple[int, int]:
    """Pick (fast, slow) maximizing train-window Sharpe."""
    best = (fast_spans[0], slow_spans[0])
    best_score = float("-inf")
    close = bars["close"].reset_index(drop=True)
    for f in fast_spans:
        for s in slow_spans:
            if f >= s:
                continue
            sig = trading_signal(close, f, s)
            res = run_backtest(bars, sig, config)
            score = sharpe_ratio(period_returns(res.equity))
            if score > best_score:
                best_score = score
                best = (f, s)
    return best


def walk_forward(
    bars: pd.DataFrame,
    fast_spans: list[int],
    slow_spans: list[int],
    train: int,
    validate: int,
    config: BacktestConfig | None = None,
) -> dict:
    """Returns out-of-sample equity curve plus per-window chosen params.

    The validate legs are run independently on their own windows (flat start,
    full initial cash each leg) and their return streams are compounded into
    one out-of-sample curve. This avoids carrying look-ahead-sensitive state
    across window boundaries.
    """
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    windows = make_windows(len(bars), train, validate)
    if not windows:
        raise ValueError("not enough bars for the requested train/validate sizes")

    oos_returns = []
    timestamps = []
    chosen = []
    for w in windows:
        train_bars = bars.iloc[w.train_start : w.train_end].reset_index(drop=True)
        val_bars = bars.iloc[w.val_start : w.val_end].reset_index(drop=True)
        fast, slow = choose_params(train_bars, fast_spans, slow_spans, config)
        chosen.append({"train_end": str(bars["timestamp"].iloc[w.train_end - 1])[:10],
                       "val_start": str(bars["timestamp"].iloc[w.val_start])[:10],
                       "fast": fast, "slow": slow})
        sig = trading_signal(val_bars["close"], fast, slow)
        res = run_backtest(val_bars, sig, config)
        rets = period_returns(res.equity)
        oos_returns.append(rets.to_numpy())
        timestamps.append(res.equity.index[1:])

    import numpy as np

    all_rets = np.concatenate(oos_returns)
    all_ts = pd.DatetimeIndex(np.concatenate(timestamps))
    equity = pd.Series((config.initial_cash if config else 1_000_000.0) * np.cumprod(1 + all_rets), index=all_ts, name="equity")
    return {"equity": equity, "returns": pd.Series(all_rets, index=all_ts), "windows": chosen}
