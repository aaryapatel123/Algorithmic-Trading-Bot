#!/usr/bin/env python3
"""Adaptive-α sweep — Pillar 2.4.

Tests dynamic EWMA smoothing where α tightens when cross-sectional noise
spikes (high CS vol → stick to prior weights) and loosens in calm markets
(low CS vol → adopt new inv-vol targets quickly). Crash override: SPY < MA200
always forces smooth_alpha_min regardless of noise level.

Configs scored on Sortino (primary) and Calmar vs. the fixed-α baseline.
Full window (2015–2026) + OOS window (2021–2026) for each config so we can
see if gains are IS-only or generalise.

    python scripts/adaptive_alpha_eval.py
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
    vol_window=20,
    abs_momentum_filter=False,
    regime_filter=False,
)

CONFIGS: list[tuple[str, dict]] = [
    ("Baseline  fixed α=0.50",
     dict(weight_smoothing=0.50)),

    ("Adaptive  min=0.10 max=0.90",
     dict(adaptive_smoothing=True, smooth_alpha_min=0.10, smooth_alpha_max=0.90)),
    ("Adaptive  min=0.20 max=0.80",
     dict(adaptive_smoothing=True, smooth_alpha_min=0.20, smooth_alpha_max=0.80)),
    ("Adaptive  min=0.30 max=0.70",
     dict(adaptive_smoothing=True, smooth_alpha_min=0.30, smooth_alpha_max=0.70)),
    ("Adaptive  min=0.10 max=0.50",
     dict(adaptive_smoothing=True, smooth_alpha_min=0.10, smooth_alpha_max=0.50)),
    ("Adaptive  min=0.20 max=0.50",
     dict(adaptive_smoothing=True, smooth_alpha_min=0.20, smooth_alpha_max=0.50)),
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
        f"\n{'Config':<32}{'CAGR':>7}{'Sortino':>9}{'Calmar':>8}"
        f"{'MaxDD':>8}{'Trades':>8}{'2020':>8}{'2021':>8}{'2022':>8}"
    )
    sep = "-" * len(hdr)

    for window_label, rows, base_sort in (
        ("FULL WINDOW (2015–2025)", full_rows, base_sort_full),
        ("OOS WINDOW  (2021–2025)", oos_rows, base_sort_oos),
    ):
        print(f"\n── {window_label} {'─'*(len(hdr)-len(window_label)-4)}")
        print(hdr)
        print(sep)
        for label, m in rows:
            beat = " ◀" if m["sortino"] > base_sort and label != rows[0][0] else ""
            print(
                f"{label:<32}{m['cagr']:>6.1f}%{m['sortino']:>9.3f}{m['calmar']:>8.3f}"
                f"{m['maxdd']:>7.1f}%{m['trades']:>8}"
                f"{m['y2020']:>7.1f}%{m['y2021']:>7.1f}%{m['y2022']:>7.1f}%{beat}"
            )
        print(sep)

    print(f"\n  Baseline Sortino — full={base_sort_full:.3f}  OOS={base_sort_oos:.3f}")
    print(
        "  Accept only if OOS Sortino also beats baseline "
        "(IS-only gains = noise, not signal)."
    )


if __name__ == "__main__":
    main()
