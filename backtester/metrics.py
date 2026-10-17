"""Risk and performance metrics. Every formula here is deterministic and
hand-checkable; tests assert them against hand-computed fixtures.

Conventions:
- Work on simple period returns r_t of the equity curve.
- N = periods per year (252 daily, 52 weekly, 12 monthly).
- Annualized return is geometric, not mean(r) * N.
- Annualized volatility and ratios scale with sqrt(N).
- Sortino denominator: downside deviation over ALL periods
  (sqrt(mean(min(r - target, 0)^2))), the more common convention; some
  implementations divide by the count of downside periods only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def period_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric: (prod(1+r))^(N/T) - 1."""
    r = returns.to_numpy(dtype=np.float64)
    if len(r) == 0:
        return 0.0
    total = np.prod(1 + r)
    if total <= 0:
        return -1.0
    return float(total ** (periods_per_year / len(r)) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Sample std (ddof=1) scaled by sqrt(N)."""
    r = returns.to_numpy(dtype=np.float64)
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def _rf_per_period(risk_free_annual: float, periods_per_year: int) -> float:
    return (1 + risk_free_annual) ** (1 / periods_per_year) - 1


def sharpe_ratio(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """(mean(r) - rf_per_period) / std(r) * sqrt(N), annualized."""
    r = returns.to_numpy(dtype=np.float64)
    if len(r) < 2:
        return 0.0
    sd = np.std(r, ddof=1)
    if sd == 0:
        return 0.0
    excess = np.mean(r) - _rf_per_period(risk_free_annual, periods_per_year)
    return float(excess / sd * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series, target_per_period: float = 0.0
) -> float:
    """sqrt(mean(min(r - target, 0)^2)) over ALL periods (population mean)."""
    r = returns.to_numpy(dtype=np.float64)
    if len(r) == 0:
        return 0.0
    downside = np.minimum(r - target_per_period, 0.0)
    return float(np.sqrt(np.mean(downside**2)))


def sortino_ratio(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """(mean(r) - rf_per_period) / downside_dev * sqrt(N), annualized.

    Downside deviation uses target = rf_per_period and the all-periods
    denominator convention (see module docstring).
    """
    r = returns.to_numpy(dtype=np.float64)
    if len(r) < 2:
        return 0.0
    rf = _rf_per_period(risk_free_annual, periods_per_year)
    dd = downside_deviation(returns, target_per_period=rf)
    if dd == 0:
        return 0.0
    excess = np.mean(r) - rf
    return float(excess / dd * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough decline, as a negative fraction (e.g. -0.23)."""
    if len(equity) == 0:
        return 0.0
    return float(drawdown_series(equity).min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch of periods spent below the running peak."""
    if len(equity) == 0:
        return 0
    below = (drawdown_series(equity) < 0).to_numpy()
    longest = run = 0
    for b in below:
        run = run + 1 if b else 0
        longest = max(longest, run)
    return longest


def win_rate(trades) -> float:
    """Fraction of round-trips (buy followed by sell) with positive P&L."""
    round_trips = 0
    wins = 0
    open_cost = None
    open_commission = 0.0
    for t in trades:
        if t.side == "buy":
            open_cost = t.value
            open_commission = t.commission
        elif t.side == "sell" and open_cost is not None:
            round_trips += 1
            pnl = t.value - t.commission - open_cost - open_commission
            wins += pnl > 0
            open_cost = None
    return wins / round_trips if round_trips else 0.0


def exposure(position: pd.Series) -> float:
    """Fraction of bars with an open position."""
    if len(position) == 0:
        return 0.0
    return float((position > 0).mean())


def summarize(result, risk_free_annual: float = 0.0) -> dict:
    """All headline metrics for a BacktestResult in one dict."""
    rets = period_returns(result.equity)
    avg_trade = (
        float(np.mean([
            (t.value - t.commission) for t in result.trades if t.side == "sell"
        ]))
        if any(t.side == "sell" for t in result.trades)
        else 0.0
    )
    return {
        "annualized_return": annualized_return(rets),
        "annualized_volatility": annualized_volatility(rets),
        "sharpe": sharpe_ratio(rets, risk_free_annual),
        "sortino": sortino_ratio(rets, risk_free_annual),
        "max_drawdown": max_drawdown(result.equity),
        "max_drawdown_duration_periods": max_drawdown_duration(result.equity),
        "win_rate": win_rate(result.trades),
        "avg_sell_value": avg_trade,
        "exposure": exposure(result.position),
        "n_trades": len(result.trades),
        "total_commission": result.total_commission,
    }
