#!/usr/bin/env python3
"""Compare covariance-aware ERC weighting against the inverse-vol baseline.

Isolates the weighting change: every config below is identical to the current
primary strategy (top-5 momentum, 6m abs-momentum filter, 25% position cap,
monthly rebalance) EXCEPT the weighting step. Reported against SPY buy & hold
across the full window and the three sub-periods from the Session-3 rubric.

Usage:
    python scripts/compare_erc.py              # full ERC-vs-inv-vol matrix
    python scripts/compare_erc.py --tilt-sweep # λ sweep (ERC↔momentum) on full period
    python scripts/compare_erc.py --quick      # single short-window smoke test
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

# Primary-strategy constants held fixed across every config (only weighting varies).
PRIMARY = dict(
    top_n=5,
    vol_window=20,
    rebalance_freq="monthly",
    regime_filter=False,
    sharpe_ranking=False,
    abs_momentum_filter=False,  # documented primary: cap does NOT stack with abs filter (SESSION_NOTES_3)
    max_weight=0.25,
)

PERIODS = [
    ("2023-2026",      datetime.date(2023, 1, 1), datetime.date(2026, 6, 1)),
    ("Full 2015-2026", datetime.date(2015, 1, 1), datetime.date(2026, 6, 1)),
]


def _row(label: str, strat: dict, bench: dict) -> str:
    def f(v, pct=False):
        if v is None:
            return "    n/a"
        return f"{v:7.2f}" if not pct else f"{v:6.1f}%"
    return (
        f"  {label:<26}"
        f"{f(strat.get('total_return_pct')):>10}"
        f"{f(strat.get('sharpe_ratio')):>9}"
        f"{f(strat.get('max_drawdown_pct')):>9}"
        f"{strat.get('total_trades', 0):>8}"
        f"{f(bench.get('total_return_pct')):>11}"
        f"{f(bench.get('sharpe_ratio')):>9}"
        f"{f(bench.get('max_drawdown_pct')):>9}"
    )


def _header(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)
    print(
        f"  {'Config':<26}{'Ret%':>10}{'Sharpe':>9}{'MaxDD':>9}{'Trades':>8}"
        f"{'SPY Ret%':>11}{'SPY Shp':>9}{'SPY DD':>9}"
    )
    print("  " + "-" * 88)


def run_config(label: str, start, end, **overrides) -> dict:
    logger.warning("Running %s | %s → %s", label, start, end)
    return run_backtest(start_date=start, end_date=end, **PRIMARY, **overrides)


def matrix() -> None:
    configs = [
        ("inv-vol (baseline)", dict(weighting="inv_vol")),
        ("ERC λ=0 (LW shrink)", dict(weighting="erc", mom_tilt=0.0)),
        ("ERC λ=0.5 (tilted)",  dict(weighting="erc", mom_tilt=0.5)),
    ]
    for pname, start, end in PERIODS:
        _header(f"{pname}  (top-5 mom + 6m abs + 25% cap; weighting varies)")
        for clabel, ov in configs:
            r = run_config(clabel, start, end, **ov)
            print(_row(clabel, r["strategy"], r["benchmark"]))
    print("=" * 92)


def tilt_sweep() -> None:
    start, end = datetime.date(2015, 1, 1), datetime.date(2026, 6, 1)
    _header("λ tilt sweep — Full 2015-2026 (ERC ↔ momentum-rank conviction)")
    base = run_config("inv-vol (baseline)", start, end, weighting="inv_vol")
    print(_row("inv-vol (baseline)", base["strategy"], base["benchmark"]))
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = run_config(f"ERC λ={lam}", start, end, weighting="erc", mom_tilt=lam)
        print(_row(f"ERC λ={lam:.2f}", r["strategy"], r["benchmark"]))
    print("=" * 92)


def quick() -> None:
    start, end = datetime.date(2022, 1, 1), datetime.date(2023, 6, 1)
    _header("QUICK smoke test — 2022-01 → 2023-06")
    for clabel, ov in [
        ("inv-vol (baseline)", dict(weighting="inv_vol")),
        ("ERC λ=0", dict(weighting="erc", mom_tilt=0.0)),
        ("ERC λ=0.5", dict(weighting="erc", mom_tilt=0.5)),
    ]:
        r = run_config(clabel, start, end, **ov)
        print(_row(clabel, r["strategy"], r["benchmark"]))
    print("=" * 92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt-sweep", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        quick()
    elif args.tilt_sweep:
        tilt_sweep()
    else:
        matrix()


if __name__ == "__main__":
    main()
