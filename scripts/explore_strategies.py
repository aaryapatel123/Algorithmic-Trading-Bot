#!/usr/bin/env python3
"""Exploration harness — backtest new strategy components and combinations.

Builds on the momentum + inverse-vol base (top-5, monthly, 158-stock universe)
and layers research-backed overlays to look for configurations that raise Sharpe
and/or cut max drawdown versus the established baseline and 6m-abs-filter optimal.

Components under test:
  * skip-month momentum (12-1): rank on return lagged by `momentum_gap` bars
  * blended momentum: rank by avg percentile across 3/6/12-month windows
  * position cap: cap any single holding via `max_weight`
  * trend overlay: partial de-risk to AGG when SPY < 200-day MA

Runs every config through the cached backtest and prints one leaderboard sorted
by Sharpe. Use --quick to run only the high-signal subset.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.utils.logging_config import setup_logging
from backtest.bt_inv_vol import run_backtest

setup_logging("WARNING")  # quiet per-run logs; we print our own table
logger = logging.getLogger(__name__)


def parse_date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


# (label, overrides) — overrides merge onto the shared base config below.
CONFIGS: list[tuple[str, dict]] = [
    # ---- references ----
    ("Baseline (no filter)",            dict()),
    ("Optimal (6m abs filter)",         dict(abs_momentum_filter=True)),
    # ---- single components ----
    ("Skip-month (12-1, gap=21)",       dict(momentum_gap=21)),
    ("Blended momentum (3/6/12m)",      dict(blended_momentum=True)),
    ("Position cap 35%",                dict(max_weight=0.35)),
    ("Position cap 25%",                dict(max_weight=0.25)),
    ("Trend overlay (derisk=0.5)",      dict(trend_overlay=True, trend_derisk=0.5)),
    ("Trend overlay (derisk=0.0)",      dict(trend_overlay=True, trend_derisk=0.0)),
    # ---- pairwise combinations ----
    ("Skip-month + cap 35%",            dict(momentum_gap=21, max_weight=0.35)),
    ("Skip-month + trend 0.5",          dict(momentum_gap=21, trend_overlay=True, trend_derisk=0.5)),
    ("Cap 35% + trend 0.5",             dict(max_weight=0.35, trend_overlay=True, trend_derisk=0.5)),
    ("Blended + cap 35%",               dict(blended_momentum=True, max_weight=0.35)),
    # ---- full stacks ----
    ("Skip + cap35 + trend0.5",         dict(momentum_gap=21, max_weight=0.35, trend_overlay=True, trend_derisk=0.5)),
    ("Skip + cap35 + trend0.5 + abs",   dict(momentum_gap=21, max_weight=0.35, trend_overlay=True, trend_derisk=0.5, abs_momentum_filter=True)),
    ("Skip + cap35 + trend0.5 + blend", dict(momentum_gap=21, max_weight=0.35, trend_overlay=True, trend_derisk=0.5, blended_momentum=True)),
]

QUICK_LABELS = {
    "Baseline (no filter)", "Optimal (6m abs filter)",
    "Skip-month (12-1, gap=21)", "Position cap 35%",
    "Trend overlay (derisk=0.5)", "Skip + cap35 + trend0.5",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy exploration sweep")
    parser.add_argument("--start", type=parse_date, default=datetime.date(2015, 1, 1))
    parser.add_argument("--end", type=parse_date, default=datetime.date.today())
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--quick", action="store_true", help="Run only the high-signal subset")
    parser.add_argument("--sort", choices=["sharpe", "return", "drawdown"], default="sharpe")
    args = parser.parse_args()

    base = dict(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
        top_n=5,
        vol_window=20,
        rebalance_freq="monthly",
        regime_filter=False,
        sharpe_ranking=False,
        abs_momentum_filter=False,
    )

    configs = [(l, o) for l, o in CONFIGS if (not args.quick or l in QUICK_LABELS)]

    rows = []
    bench = None
    for i, (label, overrides) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] running: {label} ...", file=sys.stderr, flush=True)
        result = run_backtest(**{**base, **overrides})
        s = result["strategy"]
        bench = result["benchmark"]
        rows.append({
            "label": label,
            "return": s["total_return_pct"],
            "sharpe": s["sharpe_ratio"] if s["sharpe_ratio"] is not None else -999,
            "dd": s["max_drawdown_pct"],
            "trades": s["total_trades"],
        })

    key = {"sharpe": "sharpe", "return": "return", "drawdown": "dd"}[args.sort]
    reverse = args.sort != "drawdown"
    rows.sort(key=lambda r: r[key], reverse=reverse)

    print("\n" + "=" * 88)
    print(f"  STRATEGY EXPLORATION LEADERBOARD  ({args.start} → {args.end}, sorted by {args.sort})")
    print("=" * 88)
    print(f"  {'#':>2}  {'Strategy':<34}{'Return':>11}{'Sharpe':>9}{'MaxDD':>9}{'Trades':>8}")
    print("  " + "-" * 84)
    for i, r in enumerate(rows, 1):
        star = " *" if r["label"].startswith(("Baseline", "Optimal")) else "  "
        print(f"  {i:>2}{star}{r['label']:<34}{r['return']:>10.0f}%{r['sharpe']:>9.3f}{r['dd']:>8.1f}%{r['trades']:>8}")
    print("  " + "-" * 84)
    if bench:
        print(f"      {'SPY Buy & Hold':<34}{bench['total_return_pct']:>10.0f}%"
              f"{bench['sharpe_ratio']:>9.3f}{bench['max_drawdown_pct']:>8.1f}%{'—':>8}")
    print("=" * 88)
    print("  * = reference configs from prior session   |   higher Sharpe & lower MaxDD is better")


if __name__ == "__main__":
    main()
