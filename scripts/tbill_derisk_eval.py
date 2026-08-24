#!/usr/bin/env python3
"""T-bill partial de-risk sweep — Pillar 3 first experiment.

Tests the one change the prior regime-overlay experiments never isolated:
replacing AGG (which fell -13% in 2022) with BIL (T-bills, zero duration)
as the safe-parking asset during trend-overlay de-risk periods.

Question being answered: was the binary regime filter bad, or was rotating
into *duration* bad? The trend_overlay (partial de-risk on SPY < 200MA) was
never swept against BIL — this is that test.

Configs:
  - Baseline: no overlay (the current production pick)
  - trend_overlay + BIL, derisk ∈ {0.3, 0.5, 0.7}   ← the new test
  - trend_overlay + AGG, derisk ∈ {0.3, 0.5, 0.7}    ← replicates the prior
                                                         failure mode for comparison

Scored on Sortino/Calmar (the report's committed metrics), plus calendar-year
returns for 2020 (rebound-capture check), 2021 (rotation check), 2022 (the
rate-hike year that killed the prior AGG filter).

    python scripts/tbill_derisk_eval.py
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
START = datetime.date(2015, 1, 1)
END   = datetime.date(2026, 1, 1)

BASE = dict(
    start_date=START,
    end_date=END,
    initial_cash=100_000.0,
    top_n=5,
    keep_n=8,
    max_weight=0.25,
    weight_smoothing=0.5,
    vol_window=20,
    abs_momentum_filter=False,
    regime_filter=False,
)


def metrics(res: dict) -> dict:
    tr   = res["strategy"]["total_return_pct"]
    yrs  = (END - START).days / 365.25
    cagr = ((1 + tr / 100) ** (1 / yrs) - 1) * 100

    monthly = res["strategy"]["monthly_returns"]
    mr = [r["return_pct"] for r in monthly]
    n  = len(mr)

    dd_sq = [min(r / 100, 0) ** 2 for r in mr]
    ddev  = math.sqrt(sum(dd_sq) / n * 12) if n else 0.0
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
        "cagr":    cagr,
        "sortino": sortino,
        "calmar":  calmar,
        "maxdd":   mdd * 100,
        "trades":  res["strategy"]["total_trades"],
        "y2020":   annual.get(2020, float("nan")),
        "y2021":   annual.get(2021, float("nan")),
        "y2022":   annual.get(2022, float("nan")),
    }


CONFIGS: list[tuple[str, dict]] = [
    # ── Baseline — no overlay ───────────────────────────────────────────────
    ("Baseline (no overlay)",
     dict()),

    # ── T-bill safe asset (the new test) ───────────────────────────────────
    ("BIL  derisk=0.30",
     dict(trend_overlay=True, trend_derisk=0.30, safe_asset="BIL")),
    ("BIL  derisk=0.50",
     dict(trend_overlay=True, trend_derisk=0.50, safe_asset="BIL")),
    ("BIL  derisk=0.70",
     dict(trend_overlay=True, trend_derisk=0.70, safe_asset="BIL")),

    # ── AGG safe asset (prior failure mode, for direct comparison) ──────────
    ("AGG  derisk=0.30",
     dict(trend_overlay=True, trend_derisk=0.30, safe_asset="AGG")),
    ("AGG  derisk=0.50",
     dict(trend_overlay=True, trend_derisk=0.50, safe_asset="AGG")),
    ("AGG  derisk=0.70",
     dict(trend_overlay=True, trend_derisk=0.70, safe_asset="AGG")),
]


def main() -> None:
    rows = []
    for label, extra in CONFIGS:
        res = run_backtest(**BASE, **extra)
        m   = metrics(res)
        rows.append((label, m))
        print(f"  done: {label}")

    hdr = (
        f"\n{'Config':<24}{'CAGR':>7}{'Sortino':>9}{'Calmar':>8}"
        f"{'MaxDD':>8}{'Trades':>8}{'2020':>8}{'2021':>8}{'2022':>8}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    baseline_sortino = rows[0][1]["sortino"]
    for i, (label, m) in enumerate(rows):
        beat = " ◀ BEATS BASELINE" if i > 0 and m["sortino"] > baseline_sortino else ""
        print(
            f"{label:<24}{m['cagr']:>6.1f}%{m['sortino']:>9.3f}{m['calmar']:>8.3f}"
            f"{m['maxdd']:>7.1f}%{m['trades']:>8}"
            f"{m['y2020']:>7.1f}%{m['y2021']:>7.1f}%{m['y2022']:>7.1f}%"
            f"{beat}"
        )

    print(sep)
    print(f"  window {START}→{END} | baseline Sortino={baseline_sortino:.3f}")
    print(
        "  Verdict: any BIL config that beats baseline on Sortino AND "
        "shows reduced 2022 DD = floor lever found."
    )


if __name__ == "__main__":
    main()
