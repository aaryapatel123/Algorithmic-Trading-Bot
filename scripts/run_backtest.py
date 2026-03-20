#!/usr/bin/env python3
"""CLI script to run the mean-reversion backtest across one or more symbols."""
from __future__ import annotations

import argparse
import datetime
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.utils.logging_config import setup_logging
from backtest.bt_ma_crossover import run_backtest

setup_logging("INFO")
logger = logging.getLogger(__name__)


def parse_date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def print_summary(summary: dict) -> None:
    width = 50
    print("\n" + "=" * width)
    print(f"  BACKTEST RESULTS — {summary['symbol']}")
    print("=" * width)

    sections = {
        "Period": ["start_date", "end_date"],
        "Strategy Settings": [
            "ma_period",
            "drop_threshold",
            "rsi_period",
            "rsi_buy",
            "rsi_sell",
            "trailing_stop",
        ],
        "Performance": [
            "initial_cash",
            "final_value",
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "total_trades",
            "win_rate_pct",
        ],
    }

    for section, keys in sections.items():
        print(f"\n  [{section}]")
        for key in keys:
            val = summary.get(key, "n/a")
            if isinstance(val, float):
                formatted = f"{val:.4f}" if abs(val) < 10_000 else f"{val:,.2f}"
            else:
                formatted = str(val)
            print(f"    {key:<26} {formatted}")

    print("\n" + "=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mean-reversion backtest (MA + RSI + trailing stop)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL"],
        metavar="SYMBOL",
        help="One or more stock symbols, e.g. AAPL MSFT TSLA",
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default=datetime.date.today() - datetime.timedelta(days=730),
        metavar="YYYY-MM-DD",
        help="Backtest start date",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=datetime.date.today(),
        metavar="YYYY-MM-DD",
        help="Backtest end date",
    )
    parser.add_argument("--cash", type=float, default=100_000.0, help="Initial cash")

    # Mean reversion parameters
    parser.add_argument("--ma-period", type=int, default=20, help="Moving average period")
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.03,
        help="Required drop below MA to trigger buy (0.03 = 3%%)",
    )
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period")
    parser.add_argument("--rsi-buy", type=float, default=35.0, help="RSI oversold threshold (buy signal)")
    parser.add_argument("--rsi-sell", type=float, default=65.0, help="RSI overbought threshold (sell signal)")
    parser.add_argument(
        "--trailing-stop",
        type=float,
        default=0.05,
        help="Trailing stop fraction (0.05 = 5%%)",
    )

    args = parser.parse_args()

    all_results = []
    for symbol in args.symbols:
        logger.info("Running backtest for %s …", symbol)
        try:
            summary = run_backtest(
                symbol=symbol.upper(),
                start_date=args.start,
                end_date=args.end,
                initial_cash=args.cash,
                ma_period=args.ma_period,
                drop_threshold=args.drop_threshold,
                rsi_period=args.rsi_period,
                rsi_buy=args.rsi_buy,
                rsi_sell=args.rsi_sell,
                trailing_stop=args.trailing_stop,
            )
            print_summary(summary)
            all_results.append(summary)
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", symbol, exc)

    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("  COMPARISON SUMMARY")
        print("=" * 70)
        header = f"  {'Symbol':<8} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>7} {'Win%':>7}"
        print(header)
        print("  " + "-" * 66)
        for r in all_results:
            sharpe = r["sharpe_ratio"] if r["sharpe_ratio"] is not None else float("nan")
            print(
                f"  {r['symbol']:<8} "
                f"{r['total_return_pct']:>8.2f}% "
                f"{sharpe:>8.4f} "
                f"{r['max_drawdown_pct']:>7.2f}% "
                f"{r['total_trades']:>7} "
                f"{r['win_rate_pct']:>6.1f}%"
            )
        print("=" * 70)


if __name__ == "__main__":
    main()
