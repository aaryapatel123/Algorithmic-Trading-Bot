"""Localhost web UI for the monthly rebalance signal.

Usage:
    python scripts/web_allocator.py
    # then open http://localhost:5050

Enter your current Roth IRA value to see the full allocation table
(buy / sell / hold) based on the definitive keep8 + smooth 0.5 + cap25 strategy.
"""
from __future__ import annotations

import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flask import Flask, render_template_string, request

from scripts.signal_generator import (
    CASH_BUFFER,
    STATE_FILE,
    compute_signal,
    compute_trades,
    fetch_prices,
    load_state,
    _load_dotenv,
)
from src.strategy.combined_momentum import _FALLBACK_UNIVERSE

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roth IRA Allocator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2.5rem 1rem 4rem;
  }
  h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
  .subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 2rem; }
  .card {
    background: #1e2130;
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 1.75rem;
    width: 100%;
    max-width: 680px;
    margin-bottom: 1.5rem;
  }
  .form-row { display: flex; gap: 0.75rem; align-items: flex-end; }
  label { display: block; font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.4rem; }
  input[type=number] {
    flex: 1;
    background: #0f1117;
    border: 1px solid #2d3348;
    border-radius: 8px;
    color: #e2e8f0;
    font-size: 1.1rem;
    padding: 0.65rem 0.9rem;
    outline: none;
    transition: border-color 0.15s;
  }
  input[type=number]:focus { border-color: #6366f1; }
  button {
    background: #6366f1;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.4rem;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
    white-space: nowrap;
  }
  button:hover { background: #4f46e5; }
  .hint { font-size: 0.75rem; color: #475569; margin-top: 0.5rem; }

  /* loading */
  .loading { display: none; text-align: center; padding: 1rem 0; color: #64748b; font-size: 0.9rem; }
  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid #2d3348;
    border-top-color: #6366f1;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 0.5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* results */
  .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.5rem 0 0.6rem;
  }
  .section-label:first-of-type { margin-top: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th {
    text-align: left;
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0 0.5rem 0.6rem;
    border-bottom: 1px solid #2d3348;
  }
  th.num, td.num { text-align: right; }
  td { padding: 0.55rem 0.5rem; border-bottom: 1px solid #1a1f2e; }
  tr:last-child td { border-bottom: none; }
  .pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
  }
  .pill-buy  { background: rgba(52,211,153,0.12); color: #34d399; }
  .pill-sell { background: rgba(248,113,113,0.12); color: #f87171; }
  .pill-hold { background: rgba(148,163,184,0.1);  color: #94a3b8; }
  .ticker { font-weight: 700; font-size: 0.95rem; }
  .arrow  { color: #475569; margin: 0 0.15rem; }
  .green  { color: #34d399; }
  .red    { color: #f87171; }
  .muted  { color: #64748b; }

  /* momentum table */
  .mom-rank { color: #475569; font-size: 0.8rem; }
  .selected-star { color: #6366f1; font-size: 0.7rem; margin-left: 0.2rem; }

  /* summary strip */
  .summary {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
  }
  .stat { }
  .stat-val { font-size: 1.35rem; font-weight: 700; }
  .stat-lbl { font-size: 0.72rem; color: #64748b; margin-top: 0.1rem; }

  /* status badge */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
  }
  .status-due     { background: rgba(251,191,36,0.1);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
  .status-preview { background: rgba(100,116,139,0.1); color: #64748b; border: 1px solid #2d3348; }
  .dot { width: 7px; height: 7px; border-radius: 50%; }
  .dot-due     { background: #fbbf24; }
  .dot-preview { background: #475569; }

  .footer { font-size: 0.75rem; color: #334155; margin-top: 0.75rem; text-align: center; }
</style>
</head>
<body>

<h1>Roth IRA Allocator</h1>
<p class="subtitle">keep8 · smooth 0.5 · cap25 · monthly rebalance</p>

<div class="card">
  <form method="post" id="form">
    <label for="pv">Current Roth IRA value</label>
    <div class="form-row">
      <input
        type="number"
        id="pv"
        name="portfolio_value"
        min="1"
        step="0.01"
        placeholder="e.g. 50000"
        value="{{ prefill }}"
        required
        autofocus
      >
      <button type="submit" onclick="showLoading()">Generate signal</button>
    </div>
    <p class="hint">Fetches live prices — takes ~15–25 seconds on first run.</p>
  </form>
  <div class="loading" id="loading">
    <span class="spinner"></span>Fetching live prices and computing signal…
  </div>
</div>

{% if error %}
<div class="card" style="border-color:#f87171; color:#f87171;">{{ error }}</div>
{% endif %}

{% if result %}
<div class="card">

  {% if result.is_rebalance_month %}
  <div class="status-badge status-due">
    <span class="dot dot-due"></span>Rebalance due — execute trades in Robinhood
  </div>
  {% else %}
  <div class="status-badge status-preview">
    <span class="dot dot-preview"></span>Preview only — already rebalanced this month
  </div>
  {% endif %}

  <div class="summary">
    <div class="stat">
      <div class="stat-val">${{ "{:,.0f}".format(result.portfolio_value) }}</div>
      <div class="stat-lbl">Portfolio value</div>
    </div>
    <div class="stat">
      <div class="stat-val">${{ "{:,.0f}".format(result.cash_dollars) }}</div>
      <div class="stat-lbl">Cash target (5%)</div>
    </div>
    <div class="stat">
      <div class="stat-val">{{ result.as_of }}</div>
      <div class="stat-lbl">Signal date</div>
    </div>
  </div>

  <!-- Trade list -->
  {% if result.sells %}
  <p class="section-label">Sell first</p>
  <table>
    <tr>
      <th>Ticker</th><th>Action</th><th class="num">Weight</th>
      <th class="num">Amount</th><th class="num">Shares</th><th class="num">Price</th>
    </tr>
    {% for t in result.sells %}
    <tr>
      <td class="ticker">{{ t.symbol }}</td>
      <td><span class="pill pill-sell">SELL</span></td>
      <td class="num muted">
        {{ "%.1f"|format(t.current_weight) }}%<span class="arrow">→</span><span class="red">{{ "%.1f"|format(t.target_weight) }}%</span>
      </td>
      <td class="num red">−${{ "{:,.0f}".format(t.delta_dollars|abs) }}</td>
      <td class="num">{{ t.shares if t.shares is not none else "—" }}</td>
      <td class="num muted">{{ "$%.2f"|format(t.price) if t.price else "—" }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if result.buys %}
  <p class="section-label">Then buy</p>
  <table>
    <tr>
      <th>Ticker</th><th>Action</th><th class="num">Weight</th>
      <th class="num">Amount</th><th class="num">Shares</th><th class="num">Price</th>
    </tr>
    {% for t in result.buys %}
    <tr>
      <td class="ticker">{{ t.symbol }}</td>
      <td><span class="pill pill-buy">BUY</span></td>
      <td class="num muted">
        {{ "%.1f"|format(t.current_weight) }}%<span class="arrow">→</span><span class="green">{{ "%.1f"|format(t.target_weight) }}%</span>
      </td>
      <td class="num green">+${{ "{:,.0f}".format(t.delta_dollars) }}</td>
      <td class="num">{{ t.shares if t.shares is not none else "—" }}</td>
      <td class="num muted">{{ "$%.2f"|format(t.price) if t.price else "—" }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if result.holds %}
  <p class="section-label">Hold (no change)</p>
  <table>
    <tr>
      <th>Ticker</th><th>Action</th><th class="num">Weight</th>
      <th class="num">Target value</th><th class="num">Price</th>
    </tr>
    {% for t in result.holds %}
    <tr>
      <td class="ticker">{{ t.symbol }}</td>
      <td><span class="pill pill-hold">HOLD</span></td>
      <td class="num muted">{{ "%.1f"|format(t.target_weight) }}%</td>
      <td class="num">${{ "{:,.0f}".format(t.target_dollars) }}</td>
      <td class="num muted">{{ "$%.2f"|format(t.price) if t.price else "—" }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  <!-- Final portfolio -->
  <p class="section-label">Final target portfolio</p>
  <table>
    <tr>
      <th>Ticker</th><th class="num">Weight</th><th class="num">Dollar target</th><th class="num">12-mo momentum</th>
    </tr>
    {% for row in result.final_rows %}
    <tr>
      <td class="ticker">{{ row.symbol }}</td>
      <td class="num">{{ "%.1f"|format(row.weight) }}%</td>
      <td class="num">${{ "{:,.0f}".format(row.dollars) }}</td>
      <td class="num {% if row.momentum >= 0 %}green{% else %}red{% endif %}">
        {{ "%+.1f"|format(row.momentum * 100) }}%
      </td>
    </tr>
    {% endfor %}
    <tr style="border-top: 1px solid #2d3348;">
      <td class="ticker muted">CASH</td>
      <td class="num muted">{{ "%.1f"|format(result.cash_pct) }}%</td>
      <td class="num muted">${{ "{:,.0f}".format(result.cash_dollars) }}</td>
      <td class="num muted">—</td>
    </tr>
  </table>

  <!-- Momentum ranking -->
  <p class="section-label">Top-10 momentum ranking (12-month return)</p>
  <table>
    <tr><th>#</th><th>Ticker</th><th class="num">12-mo return</th><th>Status</th></tr>
    {% for row in result.ranked_top10 %}
    <tr>
      <td class="mom-rank">{{ loop.index }}</td>
      <td class="ticker">{{ row.symbol }}</td>
      <td class="num {% if row.momentum >= 0 %}green{% else %}red{% endif %}">
        {{ "%+.1f"|format(row.momentum * 100) }}%
      </td>
      <td>
        {% if row.selected %}
        <span class="pill pill-buy">Selected</span>
        {% else %}
        <span class="muted" style="font-size:0.78rem">Watching</span>
        {% endif %}
      </td>
    </tr>
    {% endfor %}
  </table>

</div>

<p class="footer">Execute sells first · Use limit orders within the bid/ask spread · State not updated in preview mode</p>
{% endif %}

<script>
function showLoading() {
  document.getElementById('loading').style.display = 'block';
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _run_signal(portfolio_value: float) -> dict:
    _load_dotenv()
    state = load_state()
    today = datetime.date.today()
    this_month = today.strftime("%Y-%m")
    is_rebalance_month = state.get("last_rebalance") != this_month

    prices = fetch_prices(list(_FALLBACK_UNIVERSE))
    if len(prices) < 5:
        raise RuntimeError(f"Only {len(prices)} symbols returned price data — check network.")

    prev_weights = {s: v for s, v in state.get("prev_weights", {}).items() if s in prices}
    signal = compute_signal(prices, prev_weights)
    if not signal:
        raise RuntimeError("Signal computation returned empty result.")

    trades = compute_trades(signal["target_weights"], prev_weights, portfolio_value, prices)

    # Build final rows
    all_trade_map = {
        **{t["symbol"]: t for t in trades["sells"]},
        **{t["symbol"]: t for t in trades["buys"]},
        **{t["symbol"]: t for t in trades["holds"]},
    }
    final_rows = [
        {
            "symbol": sym,
            "weight": signal["target_weights"].get(sym, 0) * 100,
            "dollars": all_trade_map.get(sym, {}).get("target_dollars", 0),
            "momentum": signal["momentum"].get(sym, 0),
        }
        for sym in signal["selected"]
    ]

    ranked_top10 = [
        {
            "symbol": sym,
            "momentum": signal["ranked_momentum"][sym],
            "selected": sym in signal["selected"],
        }
        for sym in signal["ranked_top10"]
    ]

    return {
        "portfolio_value": portfolio_value,
        "cash_dollars": portfolio_value * CASH_BUFFER,
        "cash_pct": CASH_BUFFER * 100,
        "as_of": today.strftime("%B %d, %Y"),
        "is_rebalance_month": is_rebalance_month,
        "sells": trades["sells"],
        "buys": trades["buys"],
        "holds": trades["holds"],
        "final_rows": final_rows,
        "ranked_top10": ranked_top10,
    }


@app.get("/")
def index():
    state = load_state()
    prefill = state.get("portfolio_value") or ""
    return render_template_string(_PAGE, result=None, error=None, prefill=prefill)


@app.post("/")
def run():
    try:
        raw = request.form.get("portfolio_value", "").strip().replace(",", "")
        portfolio_value = float(raw)
        if portfolio_value <= 0:
            raise ValueError
    except (ValueError, TypeError):
        state = load_state()
        return render_template_string(
            _PAGE, result=None, error="Enter a valid portfolio value.", prefill=""
        ), 400

    try:
        result = _run_signal(portfolio_value)
    except Exception as exc:
        state = load_state()
        return render_template_string(
            _PAGE, result=None, error=f"Signal error: {exc}", prefill=str(portfolio_value)
        ), 500

    return render_template_string(_PAGE, result=result, error=None, prefill=str(portfolio_value))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Roth IRA Allocator at http://localhost:5050")
    print("Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=5050, debug=False)
