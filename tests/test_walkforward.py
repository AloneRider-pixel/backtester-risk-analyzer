from backtester import data as datalayer
from backtester.engine import BacktestConfig, Commission
from backtester.walkforward import choose_params, make_windows, walk_forward


def test_make_windows_tiling():
    ws = make_windows(1000, train=500, validate=250)
    assert len(ws) == 2
    assert ws[0].train_start == 0 and ws[0].val_end == 750
    assert ws[1].train_start == 250 and ws[1].val_end == 1000


def test_make_windows_rejects_short_series():
    assert make_windows(300, train=500, validate=250) == []


def test_walk_forward_returns_oos_curve():
    bars = datalayer.synthetic_bars(n=800, seed=13)
    config = BacktestConfig(slippage_bps=5.0, commission=Commission("flat", 20.0))
    out = walk_forward(bars, [5, 12], [30, 48], train=400, validate=200, config=config)
    assert len(out["windows"]) == 2
    # 2 validate legs of 200 bars each -> 199 returns per leg
    assert len(out["returns"]) == 2 * 199
    assert (out["equity"] > 0).all()
    for w in out["windows"]:
        assert w["fast"] < w["slow"]


def test_choose_params_prefers_valid_pair():
    bars = datalayer.synthetic_bars(n=300, seed=17)
    fast, slow = choose_params(bars, [5, 12], [30, 48])
    assert fast < slow
