#!/usr/bin/env python3
"""Compare overlapping rebalance tranches against the single-rebalance baseline.

Isolates the tranche change: every config is the current primary strategy
(top-5 momentum, 6m abs filter, 25% cap, inverse-vol, monthly) EXCEPT the number
of overlapping rebalance tranches. tranches=1 is the exact baseline; tranches=N
splits capital into N sub-books rebalanced on staggered days within the month and
holds their average, diversifying away rebalance-date timing luck.

Usage:
    python scripts/compare_tranches.py          # full matrix across 4 periods
    python scripts/compare_tranches.py --quick   # short-window smoke test
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.utils.logging_config import setup_logging
from backtest.bt_inv_vol import run_backtest

setup_logging("WARNING")
logger = logging.getLogger(__name__)

PRIMARY = dict(
    top_n=5,
    vol_window=20,
    rebalance_freq="monthly",
    regime_filter=False,
    sharpe_ranking=False,
    abs_momentum_filter=False,  # documented primary: cap does NOT stack with abs filter (SESSION_NOTES_3)
    max_weight=0.25,
    weighting="inv_vol",
)

PERIODS = [
    ("2023-2026",      datetime.date(2023, 1, 1), datetime.date(2026, 6, 1)),
    ("Full 2015-2026", datetime.date(2015, 1, 1), datetime.date(2026, 6, 1)),
]

CONFIGS = [
    ("1 tranche (baseline)", 1),
    ("2 tranches", 2),
    ("4 tranches", 4),
]


def _row(label: str, strat: dict, bench: dict) -> str:
    def f(v):
        return "    n/a" if v is None else f"{v:7.2f}"
    return (
        f"  {label:<24}"
        f"{f(strat.get('total_return_pct')):>10}"
        f"{f(strat.get('sharpe_ratio')):>9}"
        f"{f(strat.get('max_drawdown_pct')):>9}"
        f"{strat.get('total_trades', 0):>8}"
        f"{f(bench.get('total_return_pct')):>11}"
        f"{f(bench.get('sharpe_ratio')):>9}"
        f"{f(bench.get('max_drawdown_pct')):>9}"
    )


def _header(title: str) -> None:
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(
        f"  {'Config':<24}{'Ret%':>10}{'Sharpe':>9}{'MaxDD':>9}{'Trades':>8}"
        f"{'SPY Ret%':>11}{'SPY Shp':>9}{'SPY DD':>9}"
    )
    print("  " + "-" * 86)


def run_one(label: str, start, end, n: int) -> dict:
    logger.warning("Running %s | %s → %s", label, start, end)
    return run_backtest(start_date=start, end_date=end, **PRIMARY, tranches=n)


def matrix(periods=PERIODS) -> None:
    for pname, start, end in periods:
        _header(f"{pname}  (top-5 mom + 6m abs + 25% cap + inv-vol; tranches vary)")
        for clabel, n in CONFIGS:
            r = run_one(clabel, start, end, n)
            print(_row(clabel, r["strategy"], r["benchmark"]))
    print("=" * 90)


def quick() -> None:
    matrix([("QUICK 2022-01→2023-06", datetime.date(2022, 1, 1), datetime.date(2023, 6, 1))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick() if args.quick else matrix()


if __name__ == "__main__":
    main()
