#!/usr/bin/env python3
"""CLI script to run the MA crossover backtest."""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.utils.logging_config import setup_logging
from backtest.bt_ma_crossover import run_backtest

setup_logging("INFO")
logger = logging.getLogger(__name__)


def parse_date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MA crossover backtest")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol (default: AAPL)")
    parser.add_argument(
        "--start",
        type=parse_date,
        default=datetime.date.today() - datetime.timedelta(days=730),
        help="Start date YYYY-MM-DD (default: 2 years ago)",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=datetime.date.today(),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument("--short-ma", type=int, default=20, help="Short MA period (default: 20)")
    parser.add_argument("--long-ma", type=int, default=50, help="Long MA period (default: 50)")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Initial cash (default: 100000)")
    args = parser.parse_args()

    summary = run_backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        short_period=args.short_ma,
        long_period=args.long_ma,
        initial_cash=args.cash,
    )

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    for key, value in summary.items():
        print(f"  {key:<25} {value}")
    print("=" * 50)


if __name__ == "__main__":
    main()
