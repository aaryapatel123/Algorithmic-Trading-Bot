#!/usr/bin/env python3
"""Compare holding hysteresis + EWMA weight smoothing against the primary.

IMPLEMENTATION.md idea #3 / SESSION_NOTES_4 next-step #1: cut *turnover* without
broadening the book. Two independent, default-off knobs, both layered on the
documented primary (top-5 momentum, inv-vol, 25% cap, monthly, abs OFF):

  • keep_n           — rank buffer; retain a held name while it stays in the
                       top keep_n (book stays top_n, so no broadening).
  • weight_smoothing — EWMA blend of new inv-vol targets toward prior weights.

The bar to beat is the 0.899 inv-vol+cap25 benchmark. Judged on the SESSION_NOTES_4
rubric: robustness across BOTH 2023-2026 and full 2015-2026, turnover (trade
count), and "improves return + Sharpe together". Lower turnover should also lift
after-tax CAGR — worth weighing even when pre-tax Sharpe is a wash.

Usage:
    python scripts/compare_hysteresis.py             # full keep_n x smoothing grid
    python scripts/compare_hysteresis.py --quick     # single short-window smoke test
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

# Primary-strategy constants; only keep_n / weight_smoothing vary.
PRIMARY = dict(
    top_n=5,
    vol_window=20,
    rebalance_freq="monthly",
    regime_filter=False,
    sharpe_ranking=False,
    abs_momentum_filter=False,
    max_weight=0.25,
)

PERIODS = [
    ("2023-2026",      datetime.date(2023, 1, 1), datetime.date(2026, 6, 1)),
    ("Full 2015-2026", datetime.date(2015, 1, 1), datetime.date(2026, 6, 1)),
]

# (label, keep_n, weight_smoothing). keep_n=0 / smoothing=1.0 == off.
CONFIGS = [
    ("baseline (cap25)",        0, 1.0),
    ("hysteresis keep7",        7, 1.0),
    ("hysteresis keep8",        8, 1.0),
    ("hysteresis keep10",      10, 1.0),
    ("smoothing a=0.5",         0, 0.5),
    ("smoothing a=0.7",         0, 0.7),
    ("keep8 + smooth a=0.5",    8, 0.5),
    ("keep8 + smooth a=0.7",    8, 0.7),
]


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


def run_config(start, end, *, keep_n, weight_smoothing) -> dict:
    logger.warning("Running keep_n=%d smooth=%.2f | %s -> %s",
                   keep_n, weight_smoothing, start, end)
    return run_backtest(
        start_date=start, end_date=end,
        keep_n=keep_n, weight_smoothing=weight_smoothing,
        **PRIMARY,
    )


def grid() -> None:
    for pname, start, end in PERIODS:
        _header(f"{pname}  (hysteresis + smoothing on primary; bar = baseline = 0.899)")
        for label, keep_n, smooth in CONFIGS:
            mark = "<- bar" if (keep_n, smooth) == (0, 1.0) else ""
            r = run_config(start, end, keep_n=keep_n, weight_smoothing=smooth)
            print(_row(label, r["strategy"], r["benchmark"], mark), flush=True)
    print("=" * 88, flush=True)


def quick() -> None:
    start, end = datetime.date(2022, 1, 1), datetime.date(2023, 6, 1)
    _header("QUICK smoke test -- 2022-01 -> 2023-06")
    for label, keep_n, smooth in [
        ("baseline", 0, 1.0),
        ("hysteresis keep8", 8, 1.0),
        ("smoothing a=0.5", 0, 0.5),
        ("keep8 + smooth a=0.5", 8, 0.5),
    ]:
        r = run_config(start, end, keep_n=keep_n, weight_smoothing=smooth)
        print(_row(label, r["strategy"], r["benchmark"]), flush=True)
    print("=" * 88, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick() if args.quick else grid()


if __name__ == "__main__":
    main()
