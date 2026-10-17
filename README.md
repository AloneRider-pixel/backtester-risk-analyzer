# Automated Backtester & Portfolio Risk Analyzer

A vectorized backtesting engine and portfolio risk analyzer in Python, built
correctness-first: no look-ahead bias, realistic execution costs, formulas you
can defend line by line, and only measured numbers in this README.

**Live demo:** https://backtester-risk-analyzer.onrender.com (Render free tier; the instance sleeps when idle, so the first request after idle can take 30-60s to wake)

## Stack

- **Engine:** Python, NumPy, pandas. Vectorized signal math plus a sequential,
  stateful fill loop (see "Design" below).
- **Data:** Parquet files queried in-process with DuckDB. No database server.
- **API:** FastAPI. Backtests run in a thread pool; results cached by config hash.
- **UI:** a single-page UI served by FastAPI, using TradingView's
  lightweight-charts from a CDN for the equity and drawdown curves.
- **Tests:** 26 tests including a truncation test that catches look-ahead bias.
- **CI:** GitHub Actions (ruff + pytest + Docker build). **Docker** for deploy.

## Design: the two decisions that matter

1. **Signal generation is separated from fill simulation.** Signals are
   vectorized array math (fast, pure). Fills are sequential, because each fill
   changes cash and position and vectorizing accounting invites subtle
   look-ahead bugs. The engine iterates over plain NumPy arrays, not
   `df.iterrows()`: per-row object overhead is what makes row iteration slow.
2. **Execution rule:** the signal computed from data up to bar t-1 fills at
   bar t's open. `signals.trading_signal` shifts the raw signal by one bar;
   there is no same-bar fill anywhere in the engine. This is enforced by a
   test that truncates the data and asserts past results do not change.

## Realistic execution

- **Slippage:** basis points on the fill price (`price * (1 +/- bps/1e4)`),
  config parameter.
- **Commission:** flat fee per trade (Rs 20 discount-broker style is the
  default) or basis points of trade value.
- **Walk-forward evaluation:** rolling train/validate windows; parameters are
  chosen on train, evaluated once on unseen validate, and the validate legs
  compound into the out-of-sample curve. The in-sample optimized curve is
  noise; the walk-forward curve is the honest headline.

## Known limitations (stated, not hidden)

- The demo UI runs on **synthetic data** (seeded geometric Brownian motion).
  It has no autocorrelation, so an EMA crossover has no real edge on it; the
  demo shows the machinery, not a profitable strategy.
- `data.ingest_yfinance` can pull real daily bars, but yfinance is an
  unofficial Yahoo Finance scraper: fine for a demo, not a correctness source.
  It uses auto-adjusted close, since return math on raw close breaks around
  splits and dividends.
- Single-asset, long/flat only. No short-selling, margin, or multi-asset
  portfolios yet.
- If you backtest a stock universe defined by today's index membership,
  results carry survivorship bias. This project does not ship a point-in-time
  constituent dataset.

## Risk metrics (conventions)

Work on simple period returns `r_t`, N = periods per year (252 daily).

- Annualized return (geometric): `(prod(1+r))^(N/T) - 1`
- Annualized volatility: `std(r, ddof=1) * sqrt(N)`
- Sharpe (annualized): `(mean(r) - rf_per_period) / std(r) * sqrt(N)` with
  `rf_per_period = (1 + rf_annual)^(1/N) - 1`; rf defaults to 0
- Sortino (annualized): same numerator over downside deviation
  `sqrt(mean(min(r - rf_per_period, 0)^2))` over **all** periods (the more
  common convention; some implementations divide by downside-period count)
- Max drawdown: `min(equity / cummax(equity) - 1)`; longest drawdown duration
  reported in periods
- Plus win rate, average trade value, exposure, commission paid

## Measured performance

Measured with `benchmarks/run_benchmarks.py` (median of 5 runs) on a shared
x86-64 cloud container; rerun the script on your own hardware before quoting
these anywhere:

| Operation | 500,000 bars |
|---|---|
| Signal generation (vectorized EMA crossover) | ~12 ms |
| Full backtest incl. fill loop (pure Python) | ~390 ms |
| DuckDB range query over 500k Parquet rows | ~21 ms |

Next step if more speed is ever needed: Numba on the fill loop, added only
because a benchmark says so.

## Why Parquet + DuckDB instead of Postgres or TimescaleDB

The workload is bulk scans of immutable history: exactly what columnar storage
is built for. DuckDB queries Parquet directly, in-process, free, with zero
ops. A row-store server reads far more pages for the same scan and adds a
service to run for a workload with no concurrent writers. TimescaleDB earns
its cost for streaming ingestion, many concurrent readers, or retention
policies; a backtester has none of those. Parquet files are sorted by
(ticker, timestamp), and DuckDB's zone maps give index-like pruning. If live
market feeds were added, TimescaleDB would be worth revisiting.

## API

- `POST /backtests` — run a backtest (JSON config in, run id out; repeated
  identical configs return the cached result)
- `GET /backtests/{id}` — equity curve, drawdown, trades, metrics
- `GET /health`, `GET /` — health check and the demo UI

## Run it

```bash
pip install -r requirements-dev.txt
pytest -q                          # 26 tests
python benchmarks/run_benchmarks.py
uvicorn backtester.api:app         # UI at http://localhost:8000
```

Docker:

```bash
docker build -t backtester-risk-analyzer .
docker run -p 8000:8000 backtester-risk-analyzer
```

## Testing philosophy

The finance math is pure and deterministic, so it is tested against
hand-computed fixtures (a 5-day equity curve checked against values computed
by hand), a reference EMA implementation, cost-accounting identities
(commission paid equals fee times trades), the zero-cost buy-and-hold
identity, and the truncation test for look-ahead bias. If it is not measured
or tested, it is not claimed.

## License

MIT
