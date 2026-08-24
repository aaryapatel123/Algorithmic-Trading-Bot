#!/usr/bin/env python3
"""Joint sweep of ``top_n`` x position-cap, plus an ERC re-test at top_n=8.

Question this answers (IMPLEMENTATION.md idea #2 + SESSION_NOTES_4 next-step #3):
holding everything else at the documented primary (top-5 momentum ranking, 6m
abs filter OFF, monthly rebalance, inverse-vol weighting), does *widening the
book* (top_n 5->8) at a *tighter cap* lower idiosyncratic vol enough to beat the
0.899 inv-vol+cap25 benchmark on Sharpe -- and does covariance-aware ERC
weighting, rejected at top_n=5 because the 25% cap already de-concentrated the
book, finally earn its keep at top_n=8 where the cap binds less?

Every config is identical to the primary EXCEPT top_n, max_weight, and (for the
ERC block) the weighting step. Reported against SPY buy & hold over the two
periods in the current rubric (SESSION_NOTES_4): 2023-2026 and full 2015-2026.

Usage:
    python scripts/compare_topn_cap.py            # full grid + ERC re-test
    python scripts/compare_topn_cap.py --quick    # single short-window smoke test
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

# Primary-strategy constants held fixed; only top_n / max_weight / weighting vary.
PRIMARY = dict(
    vol_window=20,
    rebalance_freq="monthly",
    regime_filter=False,
    sharpe_ranking=False,
    abs_momentum_filter=False,  # documented primary: cap does NOT stack with abs filter
)

PERIODS = [
    ("2023-2026",      datetime.date(2023, 1, 1), datetime.date(2026, 6, 1)),
    ("Full 2015-2026", datetime.date(2015, 1, 1), datetime.date(2026, 6, 1)),
]

TOP_NS = (5, 6, 8)
CAPS = (0.15, 0.20, 0.25)
BENCHMARK = (5, 0.25)  # the 0.899 bar to beat


def _row(label: str, strat: dict, bench: dict, mark: str = "") -> str:
    def f(v):
        return "    n/a" if v is None else f"{v:7.2f}"
    return (
        f"  {label:<24}"
        f"{f(strat.get('total_return_pct')):>10}"
        f"{f(strat.get('sharpe_ratio')):>9}"
        f"{f(strat.get('max_drawdown_pct')):>9}"
        f"{strat.get('total_trades', 0):>8}"
        f"{f(bench.get('sharpe_ratio')):>9}  {mark}"
    )


def _header(title: str) -> None:
    print("\n" + "=" * 88, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 88, flush=True)
    print(
        f"  {'Config':<24}{'Ret%':>10}{'Sharpe':>9}{'MaxDD':>9}{'Trades':>8}{'SPY Shp':>9}",
        flush=True,
    )
    print("  " + "-" * 84, flush=True)


def run_config(start, end, *, top_n, max_weight, **overrides) -> dict:
    logger.warning("Running top_n=%d cap=%.2f %s | %s -> %s",
                   top_n, max_weight, overrides or "", start, end)
    return run_backtest(
        start_date=start, end_date=end,
        top_n=top_n, max_weight=max_weight,
        **PRIMARY, **overrides,
    )


def grid() -> None:
    for pname, start, end in PERIODS:
        _header(f"{pname}  (top-N x cap; inv-vol; abs OFF -- bar = top5/cap25 = 0.899)")
        for top_n in TOP_NS:
            for cap in CAPS:
                mark = "<- benchmark" if (top_n, cap) == BENCHMARK else ""
                r = run_config(start, end, top_n=top_n, max_weight=cap)
                print(_row(f"top{top_n} cap{int(cap*100)}",
                           r["strategy"], r["benchmark"], mark), flush=True)

        # ERC re-test where the cap binds least (widest book, tightest caps).
        print("  " + "-" * 84, flush=True)
        for cap in (0.15, 0.20):
            for lam in (0.0, 0.25):
                r = run_config(start, end, top_n=8, max_weight=cap,
                               weighting="erc", mom_tilt=lam)
                print(_row(f"ERC top8 cap{int(cap*100)} l={lam}",
                           r["strategy"], r["benchmark"]), flush=True)
    print("=" * 88, flush=True)


def quick() -> None:
    start, end = datetime.date(2022, 1, 1), datetime.date(2023, 6, 1)
    _header("QUICK smoke test -- 2022-01 -> 2023-06")
    for top_n in (5, 8):
        for cap in (0.20, 0.25):
            r = run_config(start, end, top_n=top_n, max_weight=cap)
            print(_row(f"top{top_n} cap{int(cap*100)}", r["strategy"], r["benchmark"]), flush=True)
    print("=" * 88, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick() if args.quick else grid()


if __name__ == "__main__":
    main()
