#!/usr/bin/env python3
"""GLD sleeve — all major windows side by side.

Runs baseline + GLD 10/15/20% across every meaningful window:
  • 2007–2026  full arc (GFC + recovery + bull + COVID + rate hike)
  • 2015–2026  main backtest window
  • 2021–2026  OOS / recent

Uses UNIVERSE_2007 (136-stock, 2004-verified) throughout so numbers are
directly comparable across windows without universe-composition drift.
The 2015-2026 Sortino will read slightly lower than the 158-stock figure
(~2.108) due to the smaller universe — that's expected.

    python scripts/sleeve_all_windows.py
"""
from __future__ import annotations

import datetime
import math
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest.bt_inv_vol import run_backtest
from scripts.stress_test_2008 import UNIVERSE_2007

RF_ANNUAL = 0.04

WINDOWS = [
    ("2007–2026", datetime.date(2007, 1, 1), datetime.date(2026, 1, 1)),
    ("2015–2026", datetime.date(2015, 1, 1), datetime.date(2026, 1, 1)),
    ("2021–2026", datetime.date(2021, 1, 1), datetime.date(2026, 1, 1)),
]

BASE = dict(
    initial_cash=100_000.0,
    stock_universe=UNIVERSE_2007,
    top_n=5,
    keep_n=8,
    max_weight=0.25,
    weight_smoothing=0.5,
    vol_window=20,
    abs_momentum_filter=False,
    regime_filter=False,
)

CONFIGS: list[tuple[str, dict]] = [
    ("Baseline",  dict()),
    ("GLD  10%",  dict(sleeve_pct=0.10, sleeve_assets=("GLD",))),
    ("GLD  15%",  dict(sleeve_pct=0.15, sleeve_assets=("GLD",))),
    ("GLD  20%",  dict(sleeve_pct=0.20, sleeve_assets=("GLD",))),
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
        "y2008": annual.get(2008, float("nan")),
        "y2020": annual.get(2020, float("nan")),
        "y2021": annual.get(2021, float("nan")),
        "y2022": annual.get(2022, float("nan")),
    }


def main() -> None:
    # Collect all results: results[window_label][config_label] = metrics dict
    all_results: dict[str, list[tuple[str, dict]]] = {}

    for win_label, start, end in WINDOWS:
        rows = []
        for cfg_label, extra in CONFIGS:
            print(f"  {win_label}  {cfg_label} ...", flush=True)
            res = run_backtest(start_date=start, end_date=end, **BASE, **extra)
            rows.append((cfg_label, metrics(res, start, end)))
        all_results[win_label] = rows

    # Print one table per window
    hdr = (
        f"{'Config':<14}{'CAGR':>7}{'Sortino':>9}{'Calmar':>8}"
        f"{'MaxDD':>8}{'2008':>8}{'2020':>8}{'2021':>8}{'2022':>8}"
    )
    sep = "-" * len(hdr)

    for win_label, _, _ in WINDOWS:
        rows = all_results[win_label]
        base_sortino = rows[0][1]["sortino"]
        base_calmar  = rows[0][1]["calmar"]
        base_maxdd   = rows[0][1]["maxdd"]
        print(f"\n── {win_label} {'─' * (len(hdr) - len(win_label) - 4)}")
        print(hdr)
        print(sep)
        for label, m in rows:
            flags = ""
            if label != rows[0][0]:
                if m["sortino"] > base_sortino: flags += " S↑"
                if m["calmar"]  > base_calmar:  flags += " C↑"
                if m["maxdd"]   < base_maxdd:   flags += " DD↓"
            y08 = f"{m['y2008']:>7.1f}%" if not math.isnan(m["y2008"]) else "     n/a"
            print(
                f"{label:<14}{m['cagr']:>6.1f}%{m['sortino']:>9.3f}{m['calmar']:>8.3f}"
                f"{m['maxdd']:>7.1f}%{y08}{m['y2020']:>7.1f}%"
                f"{m['y2021']:>7.1f}%{m['y2022']:>7.1f}%{flags}"
            )
        print(sep)

    # Scorecard: count wins across all windows
    print(f"\n── Scorecard (wins vs baseline across {len(WINDOWS)} windows) {'─'*10}")
    print(f"{'Config':<14}{'Sortino↑':>10}{'Calmar↑':>10}{'DD↓':>8}")
    print("-" * 44)
    for cfg_label, _ in CONFIGS[1:]:
        s_wins = c_wins = dd_wins = 0
        for win_label, _, _ in WINDOWS:
            rows = all_results[win_label]
            base = rows[0][1]
            m    = next(m for lbl, m in rows if lbl == cfg_label)
            if m["sortino"] > base["sortino"]: s_wins += 1
            if m["calmar"]  > base["calmar"]:  c_wins += 1
            if m["maxdd"]   < base["maxdd"]:   dd_wins += 1
        print(f"{cfg_label:<14}{s_wins:>8}/{len(WINDOWS)}{c_wins:>8}/{len(WINDOWS)}{dd_wins:>6}/{len(WINDOWS)}")
    print("-" * 44)
    print(
        "\n  *** SURVIVORSHIP BIAS WARNING ***\n"
        "  Universe (136 stocks) excludes names that failed 2004-2026.\n"
        "  Real 2008 GFC performance would have been worse for all configs."
    )


if __name__ == "__main__":
    main()
