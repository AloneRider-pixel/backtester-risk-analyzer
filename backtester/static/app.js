const $ = (id) => document.getElementById(id);

const chartOpts = {
  layout: { background: { color: "#161a20" }, textColor: "#8b929c" },
  grid: { vertLines: { color: "#1e232b" }, horzLines: { color: "#1e232b" } },
  timeScale: { borderColor: "#2c333d" },
  rightPriceScale: { borderColor: "#2c333d" },
};
const equityChart = LightweightCharts.createChart($("equity"), chartOpts);
const ddChart = LightweightCharts.createChart($("dd"), chartOpts);
let series = [];

function clearSeries() {
  series.forEach(([chart, s]) => chart.removeSeries(s));
  series = [];
}

function fmtPct(x) { return (x * 100).toFixed(2) + "%"; }

function renderMetrics(m) {
  const items = [
    ["Ann. return", fmtPct(m.annualized_return)],
    ["Ann. volatility", fmtPct(m.annualized_volatility)],
    ["Sharpe", m.sharpe.toFixed(2)],
    ["Sortino", m.sortino.toFixed(2)],
    ["Max drawdown", fmtPct(m.max_drawdown)],
    ["Longest DD", m.max_drawdown_duration_periods + " bars"],
    ["Win rate", fmtPct(m.win_rate)],
    ["Exposure", fmtPct(m.exposure)],
    ["Trades", m.n_trades],
    ["Commission paid", m.total_commission.toFixed(2)],
  ];
  $("metrics").innerHTML = items.map(([k, v]) =>
    `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

function renderTrades(trades) {
  if (!trades.length) { $("trades").innerHTML = ""; return; }
  const rows = trades.slice(-50).map(t =>
    `<tr><td>${t.t}</td><td>${t.side}</td><td>${t.qty}</td><td>${t.price}</td><td>${t.commission}</td></tr>`).join("");
  $("trades").innerHTML = `<table><thead><tr><th>Date</th><th>Side</th><th>Qty</th><th>Price</th><th>Comm.</th></tr></thead><tbody>${rows}</tbody></table>`;
}

$("run").addEventListener("click", async () => {
  $("run").disabled = true;
  $("status").textContent = "Running...";
  const body = {
    n_bars: +$("n_bars").value,
    fast: +$("fast").value,
    slow: +$("slow").value,
    slippage_bps: +$("slippage_bps").value,
    commission_type: $("commission_type").value,
    commission_value: +$("commission_value").value,
    walk_forward: $("walk_forward").value === "true",
  };
  try {
    const t0 = performance.now();
    const res = await fetch("/backtests", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const { id, cached } = await res.json();
    const data = await (await fetch(`/backtests/${id}`)).json();
    $("status").textContent = `Done in ${(performance.now() - t0).toFixed(0)} ms${cached ? " (cached result)" : ""}`;
    clearSeries();
    const s1 = equityChart.addLineSeries({ color: "#2563eb", lineWidth: 2, title: "strategy" });
    s1.setData(data.equity.map(p => ({ time: p.t, value: p.v })));
    series.push([equityChart, s1]);
    const s2 = equityChart.addLineSeries({ color: "#6b7280", lineWidth: 1, title: "buy & hold" });
    s2.setData(data.buy_and_hold.map(p => ({ time: p.t, value: p.v })));
    series.push([equityChart, s2]);
    if (data.walk_forward) {
      const s3 = equityChart.addLineSeries({ color: "#16a34a", lineWidth: 2, title: "walk-forward" });
      s3.setData(data.walk_forward.equity.map(p => ({ time: p.t, value: p.v })));
      series.push([equityChart, s3]);
      $("wf-label").textContent = " vs walk-forward out-of-sample";
    } else { $("wf-label").textContent = ""; }
    const s4 = ddChart.addAreaSeries({ lineColor: "#dc2626", topColor: "rgba(220,38,38,0.05)", bottomColor: "rgba(220,38,38,0.4)" });
    s4.setData(data.drawdown.map(p => ({ time: p.t, value: p.v })));
    series.push([ddChart, s4]);
    renderMetrics(data.metrics);
    renderTrades(data.trades);
  } catch (e) {
    $("status").textContent = "Failed: " + e.message;
  } finally {
    $("run").disabled = false;
  }
});
