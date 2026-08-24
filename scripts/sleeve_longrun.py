#!/usr/bin/env python3
"""Full-arc comparison: baseline vs GLD sleeve, 2007–2026.

Uses the 136-stock 2004-verified universe (same as stress_test_2008.py) so the
GFC period is included. Reports Sortino, Calmar, MaxDD, and key calendar years
(2008, 2020, 2022) plus the full arc CAGR.

    python scripts/sleeve_longrun.py
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
START = datetime.date(2007, 1, 1)
END   = datetime.date(2026, 1, 1)

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
    ("Baseline (no sleeve)",  dict()),
    ("GLD  10%",              dict(sleeve_pct=0.10, sleeve_assets=("GLD",))),
    ("GLD  15%",              dict(sleeve_pct=0.15, sleeve_assets=("GLD",))),
    ("GLD  20%",              dict(sleeve_pct=0.20, sleeve_assets=("GLD",))),
]


def metrics(res: dict) -> dict:
    tr   = res["strategy"]["total_return_pct"]
    yrs  = (END - START).days / 365.25
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
        "y2022": annual.get(2022, float("nan")),
    }


def main() -> None:
    rows = []
    for label, extra in CONFIGS:
        print(f"  running {label} ...")
        res = run_backtest(start_date=START, end_date=END, **BASE, **extra)
        rows.append((label, metrics(res)))

    base = rows[0][1]
    hdr = (
        f"\n{'Config':<24}{'CAGR':>7}{'Sortino':>9}{'Calmar':>8}"
        f"{'MaxDD':>8}{'Trades':>8}{'2008':>8}{'2020':>8}{'2022':>8}"
    )
    sep = "-" * len(hdr)
    print(f"\n── 2007–2026  ({len(UNIVERSE_2007)}-stock 2004-verified universe) {'─'*10}")
    print(hdr)
    print(sep)
    for label, m in rows:
        beat = " ◀" if label != rows[0][0] and m["sortino"] > base["sortino"] else ""
        dd_flag = " [DD↓]" if label != rows[0][0] and m["maxdd"] < base["maxdd"] else ""
        print(
            f"{label:<24}{m['cagr']:>6.1f}%{m['sortino']:>9.3f}{m['calmar']:>8.3f}"
            f"{m['maxdd']:>7.1f}%{m['trades']:>8}"
            f"{m['y2008']:>7.1f}%{m['y2020']:>7.1f}%{m['y2022']:>7.1f}%"
            f"{beat}{dd_flag}"
        )
    print(sep)
    print(
        "\n  *** SURVIVORSHIP BIAS WARNING ***\n"
        "  Universe excludes Lehman, Bear Stearns, Wachovia, WaMu — real 2008 was worse.\n"
        f"  Baseline 2015–2026 Sortino ≈ 2.108 (158-stock full universe)."
    )


if __name__ == "__main__":
    main()
