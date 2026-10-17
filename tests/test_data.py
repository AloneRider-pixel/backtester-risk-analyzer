import pandas as pd

from backtester import data as datalayer


def test_synthetic_bars_deterministic():
    a = datalayer.synthetic_bars(n=100, seed=1)
    b = datalayer.synthetic_bars(n=100, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()


def test_parquet_roundtrip_and_duckdb_query(tmp_path):
    df = datalayer.synthetic_bars(ticker="TEST", n=300, seed=2)
    datalayer.save_parquet(df, tmp_path)
    loaded = datalayer.query_bars("TEST", tmp_path)
    assert len(loaded) == 300
    assert loaded["timestamp"].is_monotonic_increasing
    subset = datalayer.query_bars("TEST", tmp_path, start="2020-06-01", end="2020-06-30")
    assert 0 < len(subset) < 300
    assert subset["timestamp"].min() >= pd.Timestamp("2020-06-01")
