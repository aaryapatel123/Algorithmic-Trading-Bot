#!/usr/bin/env python3
"""Compare the PDF champion strategy with and without a 60% sector cap.

Metrics reported: CAGR, Sortino, Sharpe, max drawdown, trade count.
Run from the project root:
    python backtest/bt_sector_cap_compare.py
"""
from __future__ import annotations

import datetime
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest.bt_inv_vol import run_backtest

logging.basicConfig(level=logging.WARNING)

RF_ANNUAL = 0.04
START = datetime.date(2015, 1, 1)
END = datetime.date(2026, 6, 1)
INITIAL_CASH = 100_000.0

# PDF champion configuration (keep-8 hysteresis, α=0.5 smoothing, 25% cap)
CHAMPION = dict(
    start_date=START,
    end_date=END,
    initial_cash=INITIAL_CASH,
    top_n=5,
    keep_n=8,
    weight_smoothing=0.5,
    max_weight=0.25,
    abs_momentum_filter=False,
    regime_filter=False,
    vol_window=20,
)


def sortino(monthly_rets: list[dict]) -> float:
    rf_m = RF_ANNUAL / 12
    rets = np.array([r["return_pct"] / 100.0 for r in monthly_rets])
    excess = rets - rf_m
    ann_excess = excess.mean() * 12
    neg = excess[excess < 0]
    if len(neg) == 0:
        return float("inf")
    down_dev = np.sqrt(np.mean(neg ** 2)) * np.sqrt(12)
    return ann_excess / down_dev if down_dev > 0 else float("inf")


def cagr(total_return_pct: float) -> float:
    years = (END - START).days / 365.25
    return (1 + total_return_pct / 100) ** (1 / years) - 1


def print_results(runs: list[tuple[str, dict]]) -> None:
    headers = ["Configuration", "CAGR", "Sortino", "Sharpe", "Max DD", "Trades"]
    rows: list[list[str]] = []

    for label, r in runs:
        s = r["strategy"]
        rows.append([
            label,
            f"{cagr(s['total_return_pct']) * 100:.1f}%",
            f"{sortino(s['monthly_returns']):.2f}",
            f"{s['sharpe_ratio']:.2f}" if s["sharpe_ratio"] is not None else "N/A",
            f"{s['max_drawdown_pct']:.1f}%",
            str(s["total_trades"]),
        ])

    # SPY buy-and-hold from first result's benchmark
    bm = runs[0][1]["benchmark"]
    rows.append([
        "SPY buy-and-hold",
        f"{cagr(bm['total_return_pct']) * 100:.1f}%",
        "—",
        f"{bm['sharpe_ratio']:.2f}" if bm.get("sharpe_ratio") is not None else "N/A",
        f"{bm['max_drawdown_pct']:.1f}%",
        "—",
    ])

    col_w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in col_w)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print()
    print(fmt.format(*headers))
    print(sep)
    for i, row in enumerate(rows):
        if i == len(rows) - 1:
            print(sep)
        print(fmt.format(*row))
    print()


if __name__ == "__main__":
    print(f"Backtest window: {START} → {END}")
    print(f"Universe: 158-stock fallback | top-5 | keep-8 | 25% cap | α=0.5")
    print()

    print("Running baseline (no sector cap)…")
    baseline = run_backtest(**CHAMPION, sector_cap=1.0)

    print("Running with 60% sector cap…")
    cap60 = run_backtest(**CHAMPION, sector_cap=0.60)

    print("Running with 40% sector cap…")
    cap40 = run_backtest(**CHAMPION, sector_cap=0.40)

    print_results([
        ("Baseline (no sector cap)", baseline),
        ("60% sector cap",           cap60),
        ("40% sector cap",           cap40),
    ])
