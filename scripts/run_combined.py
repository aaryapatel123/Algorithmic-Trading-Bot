#!/usr/bin/env python3
"""CLI script to run the combined 3-strategy momentum system."""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.utils.logging_config import setup_logging
from backtest.bt_combined import run_backtest

setup_logging("INFO")
logger = logging.getLogger(__name__)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt_ret(pct: float) -> str:
    """Right-aligned 6-char return string with sign."""
    if pct >= 0:
        return f"+{pct:5.1f}%"
    return f"{pct:6.1f}%"


def print_monthly_table(monthly_returns: list[dict]) -> None:
    if not monthly_returns:
        print("  (no monthly return data)")
        return

    # Group by year
    by_year: dict[int, dict[int, float]] = defaultdict(dict)
    for row in monthly_returns:
        by_year[row["year"]][row["month"]] = row["return_pct"]

    # Header
    col_w = 8
    print("\n" + "=" * (6 + col_w * 13 + 2))
    print("  MONTHLY RETURNS (%)")
    print("=" * (6 + col_w * 13 + 2))
    header = f"  {'Year':<4}  " + "".join(f"{m:>{col_w}}" for m in MONTHS) + f"{'Total':>{col_w}}"
    print(header)
    print("  " + "-" * (4 + col_w * 13 + 2))

    for year in sorted(by_year):
        months_data = by_year[year]
        # Compound annual return from monthly returns
        annual = 1.0
        row_parts = []
        for m in range(1, 13):
            if m in months_data:
                r = months_data[m]
                annual *= (1 + r / 100)
                row_parts.append(f"{_fmt_ret(r):>{col_w}}")
            else:
                row_parts.append(f"{'—':>{col_w}}")
        annual_pct = (annual - 1) * 100
        print(f"  {year:<4}  {''.join(row_parts)}{_fmt_ret(annual_pct):>{col_w}}")

    print("=" * (6 + col_w * 13 + 2))


def print_summary(result: dict) -> None:
    params = result["params"]
    strat = result["strategy"]
    bench = result["benchmark"]

    width = 58
    print("\n" + "=" * width)
    print("  COMBINED MOMENTUM STRATEGY — BACKTEST RESULTS")
    print("=" * width)

    print(f"\n  [Parameters]")
    print(f"    {'start_date':<28} {params['start_date']}")
    print(f"    {'end_date':<28} {params['end_date']}")
    print(f"    {'initial_cash':<28} {params['initial_cash']:,.0f}")
    print(f"    {'top_n (momentum stocks)':<28} {params['top_n']}")

    def _fmt(val) -> str:
        if val is None:
            return "n/a"
        if isinstance(val, float):
            return f"{val:,.4f}" if abs(val) < 10_000 else f"{val:,.2f}"
        return str(val)

    perf_rows = [
        ("final_value",       "Final Portfolio Value"),
        ("total_return_pct",  "Total Return (%)"),
        ("sharpe_ratio",      "Sharpe Ratio"),
        ("max_drawdown_pct",  "Max Drawdown (%)"),
        ("total_trades",      "Total Trades"),
        ("win_rate_pct",      "Win Rate (%)"),
    ]

    print(f"\n  {'Metric':<34} {'Strategy':>12} {'SPY B&H':>12}")
    print("  " + "-" * 58)
    for key, label in perf_rows:
        sv = strat.get(key)
        bv = bench.get(key)
        sv_s = _fmt(sv) if sv is not None else "n/a"
        bv_s = _fmt(bv) if bv is not None else "—"
        print(f"  {label:<34} {sv_s:>12} {bv_s:>12}")

    print("\n" + "=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run combined Dual Momentum + 200MA + Stock Momentum strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default=datetime.date(2015, 1, 1),
        metavar="YYYY-MM-DD",
        help="Backtest start date",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=datetime.date(2026, 3, 19),
        metavar="YYYY-MM-DD",
        help="Backtest end date",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Starting portfolio cash",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of momentum stocks to hold (equal-weighted)",
    )

    args = parser.parse_args()

    logger.info(
        "Starting combined backtest: %s → %s | cash=%.0f | top_n=%d",
        args.start, args.end, args.initial_cash, args.top_n,
    )

    result = run_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
        top_n=args.top_n,
    )

    print_summary(result)
    print_monthly_table(result["strategy"]["monthly_returns"])


if __name__ == "__main__":
    main()
