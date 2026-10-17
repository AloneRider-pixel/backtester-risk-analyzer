"""Vectorized signal generation. Pure math over NumPy/pandas, no loops.

Signals produced at bar t are only tradable at bar t+1. Every public function
here returns the *raw* signal; the shift happens exactly once, in
`trading_signal`, which is the only output the engine accepts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. pandas ewm is C-backed; do not hand-roll."""
    return close.ewm(span=span, adjust=False).mean()


def ema_crossover(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """1 when fast EMA > slow EMA else 0, computed on data up to and including bar t."""
    if fast >= slow:
        raise ValueError("fast span must be smaller than slow span")
    return (ema(close, fast) > ema(close, slow)).astype(int)


def trading_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Signal the engine may act on: raw signal shifted by one bar.

    This single shift is the main look-ahead defense at the signal layer:
    the position taken for bar t uses only information available at bar t-1.
    """
    return ema_crossover(close, fast, slow).shift(1, fill_value=0).astype(int)
