#!/usr/bin/env python3
"""Diversifying sleeve sweep — Pillar 4.

Tests a permanent allocation to defensive assets (AGG, GLD, TLT) alongside
the equity momentum book. Unlike prior regime overlays, the sleeve is always
on — no conditional logic, no regime detection. The equity budget shrinks by
sleeve_pct; the defensive assets fill that gap permanently.

Goal: reduce the 51% GFC max drawdown without sacrificing Calmar/Sortino.
Scored on Sortino + Calmar (primary) plus calendar-year returns for 2020/2021/2022.

    python scripts/sleeve_eval.py
"""
from __future__ import annotations

import datetime
import math
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest.bt_inv_vol import run_backtest

RF_ANNUAL = 0.04
FULL_START = datetime.date(2015, 1, 1)
FULL_END   = datetime.date(2026, 1, 1)
OOS_START  = datetime.date(2021, 1, 1)

BASE = dict(
    initial_cash=100_000.0,
    top_n=5,
    keep_n=8,
    max_weight=0.25,
    weight_smoothing=0.5,
    vol_window=20,
    abs_momentum_filter=False,
    regime_filter=False,
)

CONFIGS: list[tuple[str, dict]] = [
    ("Baseline (no sleeve)",
     dict()),

    # ── AGG + GLD + TLT (uncorrelated trio) ───────────────────────────────
    ("AGG+GLD+TLT  10%  equal",
     dict(sleeve_pct=0.10, sleeve_assets=("AGG", "GLD", "TLT"), sleeve_weighting="equal")),
    ("AGG+GLD+TLT  15%  equal",
     dict(sleeve_pct=0.15, sleeve_assets=("AGG", "GLD", "TLT"), sleeve_weighting="equal")),
    ("AGG+GLD+TLT  20%  equal",
     dict(sleeve_pct=0.20, sleeve_assets=("AGG", "GLD", "TLT"), sleeve_weighting="equal")),
    ("AGG+GLD+TLT  10%  inv_vol",
     dict(sleeve_pct=0.10, sleeve_assets=("AGG", "GLD", "TLT"), sleeve_weighting="inv_vol")),
    ("AGG+GLD+TLT  15%  inv_vol",
     dict(sleeve_pct=0.15, sleeve_assets=("AGG", "GLD", "TLT"), sleeve_weighting="inv_vol")),

    # ── GLD + TLT only (skip AGG — hurt badly in 2022) ────────────────────
    ("GLD+TLT  10%  equal",
     dict(sleeve_pct=0.10, sleeve_assets=("GLD", "TLT"), sleeve_weighting="equal")),
    ("GLD+TLT  15%  equal",
     dict(sleeve_pct=0.15, sleeve_assets=("GLD", "TLT"), sleeve_weighting="equal")),

    # ── GLD alone (crisis hedge, uncorrelated to equities and bonds) ───────
    ("GLD  10%",
     dict(sleeve_pct=0.10, sleeve_assets=("GLD",), sleeve_weighting="equal")),
    ("GLD  15%",
     dict(sleeve_pct=0.15, sleeve_assets=("GLD",), sleeve_weighting="equal")),
]


def metrics(res: dict, start: datetime.date, end: datetime.date) -> dict:
    tr   = res["strategy"]["total_return_pct"]
    yrs  = (end - start).days / 365.25
    cagr = ((1 + tr / 100) ** (1 / yrs) - 1) * 100

    monthly = res["strategy"]["monthly_returns"]
    mr = [r["return_pct"] for r in monthly]
    n  = len(mr)

    dd_sq   = [min(r / 100, 0) ** 2 for r in mr]
    ddev    = math.sqrt(sum(dd_sq) / n * 12) if n else 0.0
    sortino = (cagr / 100 - RF_ANNUAL) / ddev if ddev > 0 else float("inf")

    eq = [1.0]
    for r in mr:
        eq.append(eq[-1] * (1 + r / 100))
    peak, mdd = 1.0, 0.0
    for v in eq:
        peak = max(peak, v)
        mdd  = max(mdd, (peak - v) / peak)
    calmar = (cagr / 100) / mdd if mdd > 0 else float("inf")

    by_year: dict[int, float] = defaultdict(lambda: 1.0)
    for r in monthly:
        by_year[r["year"]] *= (1 + r["return_pct"] / 100)
    annual = {y: (v - 1) * 100 for y, v in by_year.items()}

    return {
        "cagr": cagr, "sortino": sortino, "calmar": calmar,
        "maxdd": mdd * 100, "trades": res["strategy"]["total_trades"],
        "y2020": annual.get(2020, float("nan")),
        "y2021": annual.get(2021, float("nan")),
        "y2022": annual.get(2022, float("nan")),
    }


def main() -> None:
    full_rows, oos_rows = [], []

    for label, extra in CONFIGS:
        print(f"  running full  {label} ...")
        res = run_backtest(start_date=FULL_START, end_date=FULL_END, **BASE, **extra)
        full_rows.append((label, metrics(res, FULL_START, FULL_END)))

        print(f"  running OOS   {label} ...")
        res = run_backtest(start_date=OOS_START, end_date=FULL_END, **BASE, **extra)
        oos_rows.append((label, metrics(res, OOS_START, FULL_END)))

    base_sort_full = full_rows[0][1]["sortino"]
    base_sort_oos  = oos_rows[0][1]["sortino"]

    hdr = (
        f"\n{'Config':<28}{'CAGR':>7}{'Sortino':>9}{'Calmar':>8}"
        f"{'MaxDD':>8}{'Trades':>8}{'2020':>8}{'2021':>8}{'2022':>8}"
    )
    sep = "-" * len(hdr)

    for window_label, rows, base_sort in (
        ("FULL WINDOW (2015–2025)", full_rows, base_sort_full),
        ("OOS WINDOW  (2021–2025)", oos_rows, base_sort_oos),
    ):
        print(f"\n── {window_label} {'─'*(max(0, len(hdr)-len(window_label)-4))}")
        print(hdr)
        print(sep)
        for label, m in rows:
            beat = " ◀" if m["sortino"] > base_sort and label != rows[0][0] else ""
            print(
                f"{label:<28}{m['cagr']:>6.1f}%{m['sortino']:>9.3f}{m['calmar']:>8.3f}"
                f"{m['maxdd']:>7.1f}%{m['trades']:>8}"
                f"{m['y2020']:>7.1f}%{m['y2021']:>7.1f}%{m['y2022']:>7.1f}%{beat}"
            )
        print(sep)

    print(f"\n  Baseline — full Sortino={base_sort_full:.3f}  OOS Sortino={base_sort_oos:.3f}")
    print(
        "  Accept if: OOS Sortino beats baseline AND MaxDD is lower.\n"
        "  Reject if: gains are IS-only or MaxDD rises."
    )


if __name__ == "__main__":
    main()
