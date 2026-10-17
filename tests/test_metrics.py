"""Hand-computed fixtures: a 5-day equity curve with values computed by hand
(spreadsheet-checkable). These catch sign errors and off-by-one shifts."""

import numpy as np
import pandas as pd
import pytest

from backtester.metrics import (
    annualized_return,
    annualized_volatility,
    downside_deviation,
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    period_returns,
    sharpe_ratio,
    sortino_ratio,
)

EQ = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0])
# returns by hand: 102/100-1, 101/102-1, 105/101-1, 103/105-1
EXPECTED_RETURNS = [0.02, -0.0098039216, 0.0396039604, -0.0190476190]


def test_period_returns_hand_computed():
    r = period_returns(EQ)
    assert np.allclose(r.to_numpy(), EXPECTED_RETURNS, atol=1e-9)


def test_annualized_return_geometric():
    # (prod(1+r))^(252/4) - 1, computed by hand from the fixture
    assert annualized_return(period_returns(EQ)) == pytest.approx(5.4379137851, abs=1e-9)


def test_annualized_volatility_sqrt_scaling():
    assert annualized_volatility(period_returns(EQ)) == pytest.approx(0.4290010018, abs=1e-9)


def test_sharpe_annualized():
    assert sharpe_ratio(period_returns(EQ)) == pytest.approx(4.5160790725, abs=1e-9)


def test_downside_deviation_all_periods_convention():
    # sqrt(mean(min(r,0)^2)) over all 4 periods, not just the 2 down periods
    assert downside_deviation(period_returns(EQ)) == pytest.approx(0.0107113103, abs=1e-9)


def test_sortino_annualized():
    assert sortino_ratio(period_returns(EQ)) == pytest.approx(11.3940199212, abs=1e-9)


def test_max_drawdown_and_duration():
    # trough 103 vs peak 105: 103/105 - 1
    assert max_drawdown(EQ) == pytest.approx(-0.0190476190, abs=1e-9)
    assert max_drawdown_duration(EQ) == 1


def test_drawdown_series_values():
    dd = drawdown_series(EQ)
    assert np.allclose(
        dd.to_numpy(), [0.0, 0.0, -0.0098039216, 0.0, -0.0190476190], atol=1e-9
    )


def test_zero_vol_returns_zero_not_nan():
    flat = pd.Series([100.0, 100.0, 100.0])
    r = period_returns(flat)
    assert sharpe_ratio(r) == 0.0
    assert annualized_volatility(r) == 0.0


def test_mdd_duration_longer_stretch():
    eq = pd.Series([100.0, 90.0, 92.0, 91.0, 110.0, 105.0])
    assert max_drawdown_duration(eq) == 3
