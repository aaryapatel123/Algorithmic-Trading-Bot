#!/usr/bin/env python3
"""Run the momentum + inverse-volatility weighting strategy.

Default (highest Sharpe): top-5 momentum stocks, inv-vol weighted, monthly
rebalance, no regime filters — 0.881 Sharpe / +1537% return (2015–2026).

Pass --compare to also run the regime-filtered version side-by-side.
Pass --regime-filter to run only the filtered version.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.utils.logging_config import setup_logging
from backtest.bt_inv_vol import run_backtest

setup_logging("INFO")
logger = logging.getLogger(__name__)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def _fmt_ret(pct: float) -> str:
    return f"+{pct:5.1f}%" if pct >= 0 else f"{pct:6.1f}%"


def _fmt(val) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:,.4f}" if abs(val) < 10_000 else f"{val:,.2f}"
    return str(val)


def print_summary(result: dict, label: str) -> None:
    p = result["params"]
    strat = result["strategy"]
    bench = result["benchmark"]
    width = 66

    print("\n" + "=" * width)
    print(f"  {label}")
    print("=" * width)
    print(f"\n  [Parameters]")
    print(f"    {'start_date':<30} {p['start_date']}")
    print(f"    {'end_date':<30} {p['end_date']}")
    print(f"    {'initial_cash':<30} {p['initial_cash']:,.0f}")
    print(f"    {'top_n':<30} {p['top_n']}")
    print(f"    {'momentum_period (days)':<30} {p['momentum_period']}")
    print(f"    {'vol_window (days)':<30} {p['vol_window']}")
    print(f"    {'rebalance_freq':<30} {p['rebalance_freq']}")
    print(f"    {'regime_filter':<30} {p['regime_filter']}")
    print(f"    {'sharpe_ranking':<30} {p.get('sharpe_ranking', False)}")
    print(f"    {'vol_targeting':<30} {p.get('vol_targeting', False)}")
    if p.get('vol_targeting'):
        print(f"    {'vol_target':<30} {p.get('vol_target', 0.15):.0%}")
        print(f"    {'vol_lookback (days)':<30} {p.get('vol_lookback', 21)}")
    print(f"    {'abs_momentum_filter':<30} {p.get('abs_momentum_filter', False)}")
    if p.get('abs_momentum_filter'):
        print(f"    {'abs_momentum_period (days)':<30} {p.get('abs_momentum_period', 126)}")
    print(f"    {'universe_size':<30} {p['universe_size']}")

    perf_rows = [
        ("final_value",      "Final Portfolio Value"),
        ("total_return_pct", "Total Return (%)"),
        ("sharpe_ratio",     "Sharpe Ratio"),
        ("max_drawdown_pct", "Max Drawdown (%)"),
        ("total_trades",     "Total Trades"),
        ("win_rate_pct",     "Win Rate (%)"),
    ]
    print(f"\n  {'Metric':<36} {'Strategy':>12} {'SPY B&H':>12}")
    print("  " + "-" * 62)
    for key, lbl in perf_rows:
        sv = strat.get(key)
        bv = bench.get(key)
        print(f"  {lbl:<36} {_fmt(sv):>12} {(_fmt(bv) if bv is not None else '—'):>12}")
    print("\n" + "=" * width)


def print_comparison(results: list[dict], labels: list[str]) -> None:
    label_w = 34
    col_w = max(14, max(len(l) + 2 for l in labels))
    width = label_w + col_w * len(labels) + 2
    print("\n" + "=" * width)
    print("  STRATEGY COMPARISON")
    print("=" * width)
    print(f"  {'Metric':<{label_w}}" + "".join(f"{l:>{col_w}}" for l in labels))
    print("  " + "-" * (width - 2))

    perf_rows = [
        ("final_value",      "Final Value"),
        ("total_return_pct", "Total Return (%)"),
        ("sharpe_ratio",     "Sharpe Ratio"),
        ("max_drawdown_pct", "Max Drawdown (%)"),
        ("total_trades",     "Total Trades"),
        ("win_rate_pct",     "Win Rate (%)"),
    ]
    for key, lbl in perf_rows:
        row = f"  {lbl:<{label_w}}"
        for r in results:
            row += f"{_fmt(r['strategy'].get(key)):>{col_w}}"
        print(row)

    bench = results[0]["benchmark"]
    print("  " + "-" * (width - 2))
    for key, lbl in [("total_return_pct", "SPY B&H Return (%)"),
                     ("sharpe_ratio",     "SPY B&H Sharpe"),
                     ("max_drawdown_pct", "SPY B&H Max DD (%)")]:
        print(f"  {lbl:<{label_w}}{_fmt(bench.get(key)):>{col_w}}")
    print("=" * width)


def print_monthly_table(result: dict, label: str) -> None:
    col_w = 8
    by_year: dict[int, dict[int, float]] = defaultdict(dict)
    for row in result["strategy"]["monthly_returns"]:
        by_year[row["year"]][row["month"]] = row["return_pct"]

    print(f"\n{'=' * (6 + col_w * 14)}")
    print(f"  MONTHLY RETURNS — {label}")
    print(f"{'=' * (6 + col_w * 14)}")
    print(f"  {'Year':<4}  " + "".join(f"{m:>{col_w}}" for m in MONTHS) + f"{'Total':>{col_w}}")
    print("  " + "-" * (4 + col_w * 13 + 2))

    for year in sorted(by_year):
        months_data = by_year[year]
        annual = 1.0
        row_parts = []
        for m in range(1, 13):
            if m in months_data:
                r = months_data[m]
                annual *= (1 + r / 100)
                row_parts.append(f"{_fmt_ret(r):>{col_w}}")
            else:
                row_parts.append(f"{'—':>{col_w}}")
        print(f"  {year:<4}  {''.join(row_parts)}{_fmt_ret((annual - 1) * 100):>{col_w}}")

    print(f"{'=' * (6 + col_w * 14)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Momentum + inv-vol weighted backtest (default: highest-Sharpe config)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", type=parse_date, default=datetime.date(2015, 1, 1), metavar="YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, default=datetime.date.today(), metavar="YYYY-MM-DD")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--weekly", action="store_true", help="Rebalance weekly (default: monthly)")
    parser.add_argument("--regime-filter", action="store_true",
                        help="Run only the regime-filtered version instead of the default")
    parser.add_argument("--sharpe-ranking", action="store_true",
                        help="Rank stocks by momentum/vol (risk-adjusted Sharpe) instead of raw 12m return")
    parser.add_argument("--compare", action="store_true",
                        help="Run both no-filter and regime-filter versions side-by-side")
    parser.add_argument("--compare-ranking", action="store_true",
                        help="Compare raw-momentum ranking vs risk-adjusted Sharpe ranking side-by-side")
    parser.add_argument("--vol-targeting", action="store_true",
                        help="Enable volatility targeting overlay (scale exposure by target_vol / realized_vol)")
    parser.add_argument("--vol-target", type=float, default=0.15, metavar="FLOAT",
                        help="Annualised target portfolio volatility for vol targeting (default: 0.15)")
    parser.add_argument("--vol-lookback", type=int, default=21, metavar="DAYS",
                        help="Trading days used to estimate realized portfolio vol (default: 21)")
    parser.add_argument("--compare-vol", action="store_true",
                        help="Compare baseline vs vol-targeting overlay side-by-side")
    parser.add_argument("--vol-sweep", action="store_true",
                        help="Sweep vol targets 15%%/20%%/25%% vs baseline in one table")
    parser.add_argument("--no-abs-momentum", dest="abs_momentum", action="store_false",
                        help="Disable the 6-month absolute momentum safety filter (enabled by default)")
    parser.set_defaults(abs_momentum=True)
    parser.add_argument("--abs-momentum-period", type=int, default=126, metavar="DAYS",
                        help="Lookback for abs momentum filter in trading days (default: 126 ≈ 6 months)")
    parser.add_argument("--compare-abs", action="store_true",
                        help="Compare baseline (no filter) vs optimal (6m abs filter) side-by-side")

    args = parser.parse_args()
    freq = "weekly" if args.weekly else "monthly"

    shared = dict(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
        top_n=args.top_n,
        vol_window=args.vol_window,
        rebalance_freq=freq,
        vol_target=args.vol_target,
        vol_lookback=args.vol_lookback,
        abs_momentum_period=args.abs_momentum_period,
    )

    if args.compare_abs:
        configs = [
            ("No Abs Filter (raw baseline)",    False),
            ("6m Abs Momentum Filter (optimal)", True),
        ]
        results, col_labels = [], []
        for label, use_abs in configs:
            logger.info("Running: %s | %s", label, freq)
            r = run_backtest(**shared, regime_filter=False, sharpe_ranking=False, abs_momentum_filter=use_abs)
            results.append(r)
            col_labels.append(label)

        print_comparison(results, col_labels)
        for r, lbl in zip(results, col_labels):
            print_monthly_table(r, lbl)

    elif args.vol_sweep:
        sweep_targets = [None, 0.15, 0.20, 0.25]
        base = {k: v for k, v in shared.items() if k not in ("vol_target", "vol_lookback")}
        results, col_labels = [], []
        for vt in sweep_targets:
            label = "Baseline" if vt is None else f"VT {vt:.0%}"
            logger.info("Running: %s | %s", label, freq)
            r = run_backtest(
                **base,
                regime_filter=False,
                sharpe_ranking=False,
                vol_targeting=(vt is not None),
                vol_target=vt if vt is not None else 0.15,
                vol_lookback=args.vol_lookback,
            )
            results.append(r)
            col_labels.append(label)

        print_comparison(results, col_labels)

    elif args.compare_vol:
        configs = [
            ("No Vol Targeting (baseline)",          False),
            (f"Vol Targeting (target={args.vol_target:.0%})", True),
        ]
        results, col_labels = [], []
        for label, use_vt in configs:
            logger.info("Running: %s | %s", label, freq)
            r = run_backtest(**shared, regime_filter=False, sharpe_ranking=False, vol_targeting=use_vt)
            results.append(r)
            col_labels.append(label)

        print_comparison(results, col_labels)
        for r, lbl in zip(results, col_labels):
            print_monthly_table(r, lbl)

    elif args.compare_ranking:
        configs = [
            ("Raw 12m Momentum Ranking",          False),
            ("Risk-Adjusted Sharpe Ranking",       True),
        ]
        results, col_labels = [], []
        for label, use_sharpe in configs:
            logger.info("Running: %s | %s", label, freq)
            r = run_backtest(**shared, regime_filter=False, sharpe_ranking=use_sharpe)
            results.append(r)
            col_labels.append(label)

        print_comparison(results, col_labels)
        for r, lbl in zip(results, col_labels):
            print_monthly_table(r, lbl)

    elif args.compare:
        configs = [
            ("Momentum + Inv-Vol (no filters)", False),
            ("Momentum + Inv-Vol + Regime Filters", True),
        ]
        results, labels = [], []
        for label, use_filter in configs:
            logger.info("Running: %s | %s", label, freq)
            r = run_backtest(**shared, regime_filter=use_filter)
            results.append(r)
            labels.append(f"{label} ({freq})")

        print_comparison(results, [f"No Filters", f"Regime Filter"])
        for r, lbl in zip(results, labels):
            print_monthly_table(r, lbl)

    else:
        use_filter = args.regime_filter
        use_sharpe = args.sharpe_ranking
        use_vt = args.vol_targeting
        use_abs = args.abs_momentum
        if use_sharpe:
            label = "Momentum + Inv-Vol — Risk-Adjusted Sharpe Ranking"
        elif use_filter:
            label = "Momentum + Inv-Vol + Regime Filters"
        elif use_vt:
            label = f"Momentum + Inv-Vol — Vol Targeting (target={args.vol_target:.0%})"
        elif use_abs:
            label = "Momentum + Inv-Vol — Optimal (6m Abs Momentum Filter)"
        else:
            label = "Momentum + Inv-Vol — No Abs Filter (raw baseline)"
        logger.info("Running: %s | %s", label, freq)
        result = run_backtest(
            **shared,
            regime_filter=use_filter,
            sharpe_ranking=use_sharpe,
            vol_targeting=use_vt,
            abs_momentum_filter=use_abs,
        )
        print_summary(result, label)
        print_monthly_table(result, label)


if __name__ == "__main__":
    main()
