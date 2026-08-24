#!/usr/bin/env python3
"""Sector concentration cap sweep — full Sortino/Calmar/CAGR/MaxDD analysis.

Tests how different sector weight ceilings affect the definitive strategy.
The cap scales down any single GICS sector whose aggregate weight exceeds the
threshold; freed weight stays as cash (no replacement with next-best names).

Definitive base config (SESSION_NOTES_12):
    top_n=5, keep_n=8, max_weight=0.25, weight_smoothing=0.5,
    abs_momentum_filter=False, sleeve_pct=0.10, sleeve_assets=("GLD",)
    → 85% equity / 10% GLD / 5% cash

Current live book: all 5 names are Information Technology (semis),
so IT sits at ~85% of portfolio. A 60% cap cuts that to 60%, routing
the freed ~25% to cash. This script measures whether that protection
earns its CAGR cost on Sortino and Calmar across both test windows.

Usage:
    .venv/bin/python scripts/compare_sector_cap.py
    .venv/bin/python scripts/compare_sector_cap.py --gfc   # add 2007-2026 window
"""
from __future__ import annotations

import argparse
import datetime
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.utils.logging_config import setup_logging
from backtest.bt_inv_vol import run_backtest

setup_logging("WARNING")

# ── Definitive config (keep all params fixed; only sector_cap varies) ────────

BASE = dict(
    top_n=5,
    keep_n=8,
    max_weight=0.25,
    weight_smoothing=0.5,
    abs_momentum_filter=False,
    regime_filter=False,
    sleeve_pct=0.10,
    sleeve_assets=("GLD",),
)

# sector_cap values to sweep (1.0 = off/baseline)
CAPS = [1.00, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]

PERIODS = [
    ("2021–2026 (OOS)",    datetime.date(2021, 1, 1), datetime.date(2026, 6, 1)),
    ("2015–2026 (full)",   datetime.date(2015, 1, 1), datetime.date(2026, 6, 1)),
]

GFC_PERIOD = ("2007–2026 (GFC)", datetime.date(2007, 1, 1), datetime.date(2026, 6, 1))

RF_ANNUAL = 0.04
MAR_MONTHLY = 0.0


# ── Metric helpers ────────────────────────────────────────────────────────────

def _sortino_calmar(monthly_rets_pct: list[float], cagr: float) -> dict:
    n = len(monthly_rets_pct)
    if n == 0:
        return {}

    dd_sq = [min(r / 100 - MAR_MONTHLY, 0) ** 2 for r in monthly_rets_pct]
    downside_dev = math.sqrt(sum(dd_sq) / n * 12)
    sortino = (cagr / 100 - RF_ANNUAL) / downside_dev if downside_dev > 0 else float("inf")

    equity = [1.0]
    for r in monthly_rets_pct:
        equity.append(equity[-1] * (1 + r / 100))
    peak = 1.0
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak)

    calmar = (cagr / 100) / max_dd if max_dd > 0 else float("inf")
    neg = sum(1 for r in monthly_rets_pct if r < 0)

    return {
        "sortino":    round(sortino, 3),
        "downside_dev": round(downside_dev * 100, 2),
        "max_dd":     round(max_dd * 100, 2),
        "calmar":     round(calmar, 3),
        "neg_months": neg,
    }


def _run(cap: float, start: datetime.date, end: datetime.date) -> dict:
    result = run_backtest(
        start_date=start, end_date=end,
        sector_cap=cap,
        **BASE,
    )
    s = result["strategy"]
    total_ret = s["total_return_pct"]
    years = (end - start).days / 365.25
    cagr = ((1 + total_ret / 100) ** (1 / years) - 1) * 100

    monthly_rets = [r["return_pct"] for r in s["monthly_returns"]]
    metrics = _sortino_calmar(monthly_rets, cagr)

    return {
        "cagr":    round(cagr, 1),
        "sharpe":  s["sharpe_ratio"],
        "trades":  s["total_trades"],
        **metrics,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def _label(cap: float) -> str:
    if cap >= 1.0:
        return "no cap (baseline)"
    return f"sector cap {int(cap*100)}%"


def _print_table(period_name: str, rows: list[tuple[float, dict]]) -> None:
    W = 105
    print()
    print("=" * W)
    print(f"  {period_name}")
    print("=" * W)
    print(
        f"  {'Config':<22} | {'CAGR':>7} | {'Sortino':>8} | {'Calmar':>7} | "
        f"{'MaxDD':>7} | {'DwnDev':>7} | {'NegMo':>6} | {'Sharpe':>7} | {'Trades':>7}"
    )
    print("  " + "-" * (W - 2))

    baseline_cagr   = rows[0][1]["cagr"]
    baseline_sortino = rows[0][1]["sortino"]
    baseline_calmar  = rows[0][1]["calmar"]
    baseline_maxdd   = rows[0][1]["max_dd"]

    for cap, r in rows:
        label = _label(cap)
        dcagr  = f"({r['cagr'] - baseline_cagr:+.1f})" if cap < 1.0 else ""
        dsort  = f"({r['sortino'] - baseline_sortino:+.3f})" if cap < 1.0 else ""
        dcal   = f"({r['calmar'] - baseline_calmar:+.3f})" if cap < 1.0 else ""
        ddd    = f"({r['max_dd'] - baseline_maxdd:+.1f})" if cap < 1.0 else ""
        mark   = " ← baseline" if cap >= 1.0 else ""

        sharpe_str = f"{r['sharpe']:.3f}" if r["sharpe"] is not None else "  N/A"
        print(
            f"  {label:<22} | {r['cagr']:>5.1f}% {dcagr:<7}| {r['sortino']:>6.3f} {dsort:<8}"
            f"| {r['calmar']:>5.3f} {dcal:<7}| {r['max_dd']:>5.1f}% {ddd:<6}"
            f"| {r['downside_dev']:>5.2f}%  | {r['neg_months']:>5}  | {sharpe_str:>7} | {r['trades']:>7}{mark}"
        )

    print("=" * W)

    # Highlight any cap that beats baseline on Sortino AND Calmar AND lowers MaxDD
    winners = [
        (cap, r) for cap, r in rows[1:]
        if r["sortino"] >= baseline_sortino and r["calmar"] >= baseline_calmar and r["max_dd"] <= baseline_maxdd
    ]
    if winners:
        print()
        print("  ★ Configs that match/beat baseline on Sortino, Calmar, AND MaxDD:")
        for cap, r in winners:
            print(f"    sector cap {int(cap*100)}%  →  "
                  f"Sortino {r['sortino']:.3f} / Calmar {r['calmar']:.3f} / MaxDD {r['max_dd']:.1f}%")
    else:
        print()
        print("  ✗ No sector cap config beats baseline on Sortino + Calmar + MaxDD simultaneously.")
        best_sortino = max(rows[1:], key=lambda x: x[1]["sortino"])
        best_calmar  = max(rows[1:], key=lambda x: x[1]["calmar"])
        best_dd      = min(rows[1:], key=lambda x: x[1]["max_dd"])
        print(f"    Best Sortino : sector cap {int(best_sortino[0]*100)}%  → {best_sortino[1]['sortino']:.3f}")
        print(f"    Best Calmar  : sector cap {int(best_calmar[0]*100)}%   → {best_calmar[1]['calmar']:.3f}")
        print(f"    Best MaxDD   : sector cap {int(best_dd[0]*100)}%   → {best_dd[1]['max_dd']:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(include_gfc: bool = False) -> None:
    periods = list(PERIODS)
    if include_gfc:
        periods.append(GFC_PERIOD)

    print()
    print("Sector Concentration Cap Sweep")
    print("Base config: keep8 + smooth0.5 + cap25 + GLD10% (definitive pick)")
    print(f"Caps tested: {[int(c*100) for c in CAPS]}%  (100% = off)")
    print()

    for pname, start, end in periods:
        print(f"Running {pname} ...", flush=True)
        rows: list[tuple[float, dict]] = []
        for cap in CAPS:
            label = _label(cap)
            print(f"  {label} ...", flush=True)
            rows.append((cap, _run(cap, start, end)))
        _print_table(pname, rows)

    print()
    print("Notes:")
    print("  · Freed weight stays as cash — no replacement with next-best names.")
    print("  · Current live book: AMAT/LRCX/AMD/INTC/MU — all Information Technology.")
    print("  · IT sits at ~85% of portfolio with no cap (equity budget = 85%).")
    print("  · A 60% cap reduces IT to 60%, routing ~25pp to cash.")
    print("  · MaxDD figures computed from monthly equity curve (lower than daily).")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfc", action="store_true", help="Include the 2007-2026 GFC window")
    args = ap.parse_args()
    main(include_gfc=args.gfc)
