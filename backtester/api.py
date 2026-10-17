"""FastAPI layer: run backtests, serve results as JSON, serve the demo UI.

Backtests run in a thread pool so a slow run does not block the event loop.
Results are cached in memory keyed by a hash of the run config plus the data
version, so repeated identical requests are free (reproducibility is a feature).
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data as datalayer
from .engine import BacktestConfig, Commission, buy_and_hold, run_backtest
from .metrics import drawdown_series, summarize
from .signals import trading_signal
from .walkforward import walk_forward

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Backtester & Portfolio Risk Analyzer", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# run_id -> stored result dict (in-memory; a demo, not a store)
RUNS: dict[str, dict] = {}
# config hash -> run_id (result cache)
CACHE: dict[str, str] = {}


class BacktestRequest(BaseModel):
    ticker: str = "SYNTH"
    n_bars: int = Field(default=1500, ge=100, le=2_000_000)
    seed: int = 42
    fast: int = Field(default=12, ge=2)
    slow: int = Field(default=48, ge=3)
    initial_cash: float = 1_000_000.0
    slippage_bps: float = Field(default=5.0, ge=0.0)
    commission_type: str = "flat"  # "flat" | "bps"
    commission_value: float = Field(default=20.0, ge=0.0)
    risk_free_annual: float = 0.0
    walk_forward: bool = False
    train: int = Field(default=504, ge=50)
    validate: int = Field(default=126, ge=20)


def _cache_key(req: BacktestRequest) -> str:
    return hashlib.sha256(req.model_dump_json().encode()).hexdigest()[:16]


def _execute(req: BacktestRequest) -> dict:
    bars = datalayer.synthetic_bars(ticker=req.ticker, n=req.n_bars, seed=req.seed)
    config = BacktestConfig(
        initial_cash=req.initial_cash,
        slippage_bps=req.slippage_bps,
        commission=Commission(type=req.commission_type, value=req.commission_value),
    )
    signal = trading_signal(bars["close"], req.fast, req.slow)
    result = run_backtest(bars, signal, config)
    bh = buy_and_hold(bars, req.initial_cash)
    out = {
        "metrics": summarize(result, req.risk_free_annual),
        "buy_and_hold_metrics": {
            "annualized_return": float((bh.iloc[-1] / bh.iloc[0]) ** (252 / (len(bh) - 1)) - 1),
        },
        "equity": [
            {"t": str(ts)[:10], "v": round(float(v), 2)} for ts, v in result.equity.items()
        ],
        "buy_and_hold": [
            {"t": str(ts)[:10], "v": round(float(v), 2)} for ts, v in bh.items()
        ],
        "drawdown": [
            {"t": str(ts)[:10], "v": round(float(v), 4)}
            for ts, v in drawdown_series(result.equity).items()
        ],
        "trades": [
            {
                "t": str(t.timestamp)[:10],
                "side": t.side,
                "qty": round(t.quantity, 4),
                "price": round(t.price, 2),
                "commission": round(t.commission, 2),
            }
            for t in result.trades
        ],
        "config": req.model_dump(),
    }
    if req.walk_forward:
        wf = walk_forward(
            bars,
            fast_spans=[5, 10, 12, 20],
            slow_spans=[30, 48, 60, 100],
            train=req.train,
            validate=req.validate,
            config=config,
        )
        out["walk_forward"] = {
            "equity": [
                {"t": str(ts)[:10], "v": round(float(v), 2)} for ts, v in wf["equity"].items()
            ],
            "windows": wf["windows"],
        }
    return out


@app.post("/backtests")
async def create_backtest(req: BacktestRequest):
    key = _cache_key(req)
    if key in CACHE and CACHE[key] in RUNS:
        return {"id": CACHE[key], "cached": True}
    result = await run_in_threadpool(_execute, req)
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = result
    CACHE[key] = run_id
    return {"id": run_id, "cached": False}


@app.get("/backtests/{run_id}")
async def get_backtest(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="unknown run id")
    return RUNS[run_id]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
