"""Data layer: synthetic bars, Parquet cache, DuckDB queries, optional yfinance ingest.

Design note (see README): the workload is bulk scans of immutable history, so we
use columnar Parquet files queried in-process with DuckDB instead of running a
database server. Parquet is sorted by (ticker, timestamp); DuckDB zone maps then
prune row groups the same way a composite index would.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path("data_cache")

COLUMNS = ["ticker", "timestamp", "open", "high", "low", "close", "volume"]


def synthetic_bars(
    ticker: str = "SYNTH",
    n: int = 1500,
    seed: int = 42,
    start: str = "2020-01-01",
    s0: float = 100.0,
    mu: float = 0.0004,
    sigma: float = 0.015,
) -> pd.DataFrame:
    """Deterministic geometric-Brownian-motion daily bars for demos and tests.

    This is synthetic data: it carries no market information. It exists so the
    demo and the test-suite never depend on a network or a data vendor.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    close = s0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = s0
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.001, n - 1))
    spread = np.abs(rng.normal(0.004, 0.002, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(100_000, 5_000_000, n).astype(np.int64)
    # daily spacing in datetime64[ms]: 500k bars overflow datetime64[ns]
    timestamps = np.datetime64(start) + np.arange(n) * np.timedelta64(1, "D")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def save_parquet(df: pd.DataFrame, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path:
    """Write bars to data_cache/<ticker>.parquet, sorted by (ticker, timestamp)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ticker = str(df["ticker"].iloc[0])
    out = cache_dir / f"{ticker}.parquet"
    df.sort_values(["ticker", "timestamp"]).to_parquet(out, index=False)
    return out


def load_parquet(ticker: str, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    path = Path(cache_dir) / f"{ticker}.parquet"
    return pd.read_parquet(path)


def query_bars(
    ticker: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Query one ticker's bars through DuckDB over the Parquet file."""
    path = Path(cache_dir) / f"{ticker}.parquet"
    clauses = [f"ticker = '{ticker}'"]
    if start:
        clauses.append(f"timestamp >= TIMESTAMP '{start}'")
    if end:
        clauses.append(f"timestamp <= TIMESTAMP '{end} 23:59:59'")
    where = " AND ".join(clauses)
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{path}') WHERE {where} ORDER BY timestamp"
    ).df()


def cache_key(ticker: str, start: str | None, end: str | None) -> str:
    """Content key used to name cached downloads reproducibly."""
    raw = f"{ticker}|{start}|{end}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def ingest_yfinance(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> Path:
    """Download daily bars with yfinance and cache as Parquet.

    yfinance is an unofficial Yahoo Finance scraper. Use it for demos, not as a
    correctness source. We use auto_adjust=True so close is split- and
    dividend-adjusted; return math on raw close breaks around corporate actions.
    """
    import yfinance as yf  # optional dependency, imported lazily

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower).reset_index()
    df = df.rename(columns={"date": "timestamp", "datetime": "timestamp"})
    df["ticker"] = ticker
    df = df[COLUMNS].dropna(subset=["close"])
    return save_parquet(df, cache_dir)
