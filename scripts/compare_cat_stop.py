#!/usr/bin/env python3
"""Catastrophic stop sweep — daily per-name stop combined with 80% sector cap.

Tests how a hard intra-month stop (exit any held name that falls >X% below its
average cost basis on any daily close) interacts with the definitive strategy.
The stop fires between monthly rebalances and excludes the stopped name from
re-selection until the next rebalance period.

Combined with the 80% sector cap (near-zero normal-market cost: −0.3pp CAGR)
this targets the specific tail risk: all 5 names in one sector (currently all
IT/semis) suffer a sharp intra-month collapse.

Base config (definitive pick + light sector cap):
    top_n=5, keep_n=8, max_weight=0.25, weight_smoothing=0.5,
    abs_momentum_filter=False, sleeve_pct=0.10, sleeve_assets=("GLD",),
    sector_cap=0.80

Usage:
    .venv/bin/python scripts/compare_cat_stop.py
    .venv/bin/python scripts/compare_cat_stop.py --gfc   # add 2007-2026 window
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

BASE = dict(
    top_n=5,
    keep_n=8,
    max_weight=0.25,
    weight_smoothing=0.5,
    abs_momentum_filter=False,
    regime_filter=False,
    sleeve_pct=0.10,
    sleeve_assets=("GLD",),
    sector_cap=0.80,
)

# (label, cat_stop_pct)
CONFIGS = [
    ("no stop + 80% cap (base)", 0.00),
    ("stop 15%",                 0.15),
    ("stop 20%",                 0.20),
    ("stop 25%",                 0.25),
    ("stop 30%",                 0.30),
]

# Also compare against the definitive with no sector cap (for full context)
PURE_BASELINE = dict(
    top_n=5, keep_n=8, max_weight=0.25, weight_smoothing=0.5,
    abs_momentum_filter=False, regime_filter=False,
    sleeve_pct=0.10, sleeve_assets=("GLD",),
    sector_cap=1.0,
)

PERIODS = [
    ("2021–2026 (OOS)",   datetime.date(2021, 1, 1), datetime.date(2026, 6, 1)),
    ("2015–2026 (full)",  datetime.date(2015, 1, 1), datetime.date(2026, 6, 1)),
]

GFC_PERIOD = ("2007–2026 (GFC)", datetime.date(2007, 1, 1), datetime.date(2026, 6, 1))

RF_ANNUAL = 0.04
MAR_MONTHLY = 0.0


def _sortino_calmar(monthly_rets: list[float], cagr: float) -> dict:
    n = len(monthly_rets)
    if n == 0:
        return {}
    dd_sq = [min(r / 100 - MAR_MONTHLY, 0) ** 2 for r in monthly_rets]
    downside_dev = math.sqrt(sum(dd_sq) / n * 12)
    sortino = (cagr / 100 - RF_ANNUAL) / downside_dev if downside_dev > 0 else float("inf")

    equity = [1.0]
    for r in monthly_rets:
        equity.append(equity[-1] * (1 + r / 100))
    peak, max_dd = 1.0, 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak)

    calmar = (cagr / 100) / max_dd if max_dd > 0 else float("inf")
    neg = sum(1 for r in monthly_rets if r < 0)
    return {
        "sortino":  round(sortino, 3),
        "max_dd":   round(max_dd * 100, 2),
        "calmar":   round(calmar, 3),
        "neg_mo":   neg,
    }


def _run(params: dict, cat_stop: float, start: datetime.date, end: datetime.date) -> dict:
    result = run_backtest(start_date=start, end_date=end, cat_stop_pct=cat_stop, **params)
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
        "stops":   s.get("cat_stop_triggers", 0),
        **metrics,
    }


def _print_table(period_name: str, rows: list[tuple[str, dict]], baseline_row: dict) -> None:
    W = 115
    print()
    print("=" * W)
    print(f"  {period_name}")
    print("=" * W)
    print(
        f"  {'Config':<28} | {'CAGR':>7} | {'Sortino':>8} | {'Calmar':>7} | "
        f"{'MaxDD':>7} | {'NegMo':>6} | {'Sharpe':>7} | {'Trades':>7} | {'Stops':>6}"
    )
    print("  " + "-" * (W - 2))

    # Print the clean definitive baseline (no cap, no stop) first for context
    br = baseline_row
    print(
        f"  {'definitive (no cap/stop)':<28} | {br['cagr']:>5.1f}%  | {br['sortino']:>8.3f} | "
        f"{br['calmar']:>7.3f} | {br['max_dd']:>5.1f}%  | {br['neg_mo']:>6} | "
        f"{br['sharpe'] or 0.0:>7.3f} | {br['trades']:>7} | {'—':>6}  ← reference"
    )
    print("  " + "·" * (W - 2))

    # All sector-cap + stop configs
    base = rows[0][1]  # "no stop + 80% cap" is the comparison base
    for label, r in rows:
        is_base = label == rows[0][0]
        tag = " ← base" if is_base else ""
        dcagr   = "" if is_base else f"({r['cagr'] - base['cagr']:+.1f})"
        dsort   = "" if is_base else f"({r['sortino'] - base['sortino']:+.3f})"
        dcal    = "" if is_base else f"({r['calmar'] - base['calmar']:+.3f})"
        ddd     = "" if is_base else f"({r['max_dd'] - base['max_dd']:+.1f})"
        sharpe_str = f"{r['sharpe']:.3f}" if r["sharpe"] is not None else "  N/A"
        stops_str = str(r["stops"]) if r["stops"] else "0"
        print(
            f"  {label:<28} | {r['cagr']:>5.1f}% {dcagr:<5} | {r['sortino']:>6.3f} {dsort:<8}"
            f"| {r['calmar']:>5.3f} {dcal:<7}| {r['max_dd']:>5.1f}% {ddd:<5}"
            f"| {r['neg_mo']:>6} | {sharpe_str:>7} | {r['trades']:>7} | {stops_str:>5}{tag}"
        )
    print("=" * W)

    # Verdict
    winners = [
        (label, r) for label, r in rows[1:]  # skip base
        if r["sortino"] >= base["sortino"] and r["calmar"] >= base["calmar"]
    ]
    print()
    if winners:
        print("  ★ Stop configs that match/beat the 80%-cap base on BOTH Sortino AND Calmar:")
        for label, r in winners:
            print(f"    {label}  →  Sortino {r['sortino']:.3f} ({r['sortino']-base['sortino']:+.3f}) | "
                  f"Calmar {r['calmar']:.3f} ({r['calmar']-base['calmar']:+.3f}) | "
                  f"MaxDD {r['max_dd']:.1f}% ({r['max_dd']-base['max_dd']:+.1f}pp) | "
                  f"Stops fired: {r['stops']}")
    else:
        # Find the least bad tradeoff
        best_sort = max(rows[1:], key=lambda x: x[1]["sortino"])
        best_cal  = max(rows[1:], key=lambda x: x[1]["calmar"])
        best_dd   = min(rows[1:], key=lambda x: x[1]["max_dd"])
        print("  ✗ No stop config beats the 80%-cap base on Sortino + Calmar simultaneously.")
        print(f"    Best Sortino : {best_sort[0]}  → {best_sort[1]['sortino']:.3f}  "
              f"({best_sort[1]['sortino']-base['sortino']:+.3f}), stops={best_sort[1]['stops']}")
        print(f"    Best Calmar  : {best_cal[0]}  → {best_cal[1]['calmar']:.3f}  "
              f"({best_cal[1]['calmar']-base['calmar']:+.3f}), stops={best_cal[1]['stops']}")
        print(f"    Best MaxDD   : {best_dd[0]}  → {best_dd[1]['max_dd']:.1f}%  "
              f"({best_dd[1]['max_dd']-base['max_dd']:+.1f}pp), stops={best_dd[1]['stops']}")


def main(include_gfc: bool = False) -> None:
    periods = list(PERIODS)
    if include_gfc:
        periods.append(GFC_PERIOD)

    print()
    print("Catastrophic Stop Sweep (+ 80% sector cap)")
    print("Base: keep8 + smooth0.5 + cap25 + GLD10% + sector_cap=0.80")
    print(f"Stop thresholds: {[int(c[1]*100) for c in CONFIGS[1:]]}% (0% = off)")
    print()

    for pname, start, end in periods:
        print(f"Running {pname} ...", flush=True)

        # Clean definitive reference (no sector cap, no stop)
        print("  reference (no cap/stop) ...", flush=True)
        baseline_row = _run(PURE_BASELINE, 0.0, start, end)

        rows: list[tuple[str, dict]] = []
        for label, cat_stop in CONFIGS:
            print(f"  {label} ...", flush=True)
            rows.append((label, _run(BASE, cat_stop, start, end)))

        _print_table(pname, rows, baseline_row)

    print()
    print("Notes:")
    print("  · Stop fires daily (every bar), not just at rebalance dates.")
    print("  · Avg cost basis (backtrader position.price) is the stop reference.")
    print("  · Stopped names are excluded from reselection until the next month's rebalance.")
    print("  · 80% sector cap: costs ~−0.3pp CAGR vs no cap; frees 5pp to cash when all 5 names are IT.")
    print("  · Stops count: total intra-month exits across the full backtest period.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfc", action="store_true", help="Include 2007-2026 GFC window")
    args = ap.parse_args()
    main(include_gfc=args.gfc)
