"""Fill simulation: sequential, stateful, honest.

Signals are vectorized (see signals.py); fills are not. Each fill changes cash
and position, so the loop cannot be vectorized without inviting look-ahead
bugs. We iterate over plain NumPy arrays, not DataFrame rows: per-row object
overhead is what makes `df.iterrows()` slow, not Python itself.

Execution rule (documented in the README): the signal computed from data up to
bar t-1 is filled at bar t's open. There is no same-bar fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Commission:
    """flat: fixed fee per trade (e.g. Rs 20 discount-broker style).
    bps: basis points of trade value."""

    type: str = "flat"  # "flat" | "bps"
    value: float = 20.0

    def cost(self, trade_value: float) -> float:
        if self.type == "flat":
            return self.value
        if self.type == "bps":
            return trade_value * self.value / 1e4
        raise ValueError(f"unknown commission type: {self.type}")


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    slippage_bps: float = 5.0
    commission: Commission = field(default_factory=Commission)
    fill_at: str = "open"  # fills happen at next bar's open after the signal


@dataclass
class Trade:
    timestamp: pd.Timestamp
    side: str  # "buy" | "sell"
    quantity: float
    price: float  # after slippage
    commission: float
    value: float  # quantity * price


@dataclass
class BacktestResult:
    equity: pd.Series  # daily portfolio value, indexed by timestamp
    trades: list[Trade]
    position: pd.Series
    config: BacktestConfig

    @property
    def total_commission(self) -> float:
        return sum(t.commission for t in self.trades)


def run_backtest(
    bars: pd.DataFrame, signal: pd.Series, config: BacktestConfig | None = None
) -> BacktestResult:
    """All-in long/flat backtest over one asset.

    bars: DataFrame with timestamp/open/close (sorted by timestamp).
    signal: Series aligned to bars, values in {0, 1}; signal[i] is the target
        position for bar i and must already be shifted (use signals.trading_signal).
    Fill price: bar open adjusted by slippage (buys pay more, sells receive less).
    """
    config = config or BacktestConfig()
    if config.fill_at != "open":
        raise ValueError("only fill_at='open' is implemented")
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    sig = signal.to_numpy(dtype=np.float64)
    if len(sig) != len(bars):
        raise ValueError("signal length must match bars length")

    opens = bars["open"].to_numpy(dtype=np.float64)
    closes = bars["close"].to_numpy(dtype=np.float64)
    timestamps = pd.to_datetime(bars["timestamp"]).reset_index(drop=True)

    n = len(bars)
    cash = config.initial_cash
    qty = 0.0
    equity = np.empty(n)
    position = np.empty(n)
    trades: list[Trade] = []
    slip = config.slippage_bps / 1e4

    for i in range(n):
        target = sig[i]
        if target == 1.0 and qty == 0.0:
            price = opens[i] * (1 + slip)
            # size so cash covers the purchase plus the commission on it
            if config.commission.type == "flat":
                qty = (cash - config.commission.value) / price if cash > config.commission.value else 0.0
            else:
                qty = cash / (price * (1 + config.commission.value / 1e4))
            if qty > 0:
                value = qty * price
                cost = config.commission.cost(value)
                cash -= value + cost
                trades.append(Trade(timestamps[i], "buy", qty, price, cost, value))
        elif target == 0.0 and qty > 0.0:
            price = opens[i] * (1 - slip)
            value = qty * price
            cost = config.commission.cost(value)
            cash += value - cost
            trades.append(Trade(timestamps[i], "sell", qty, price, cost, value))
            qty = 0.0
        position[i] = qty
        equity[i] = cash + qty * closes[i]

    return BacktestResult(
        equity=pd.Series(equity, index=timestamps, name="equity"),
        trades=trades,
        position=pd.Series(position, index=timestamps, name="position"),
        config=config,
    )


def buy_and_hold(bars: pd.DataFrame, initial_cash: float = 1_000_000.0) -> pd.Series:
    """Reference equity curve: fully invested at the first bar's open, no costs."""
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    qty = initial_cash / bars["open"].iloc[0]
    ts = pd.to_datetime(bars["timestamp"]).reset_index(drop=True)
    return pd.Series(qty * bars["close"].to_numpy(), index=ts, name="equity")
