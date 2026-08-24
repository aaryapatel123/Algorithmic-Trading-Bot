#!/usr/bin/env python3
"""Recession-rotation sweep: compare baseline vs consumer-defensive rotation.

Tests whether switching to top-momentum consumer defensives (Staples + Healthcare
+ Utilities) during GDP-defined recessions improves risk-adjusted returns.

Recession signal: 2 consecutive negative QoQ real GDP quarters → entry;
1 positive QoQ quarter → exit. 1-month publication lag (advance estimate).

Uses UNIVERSE_2007 (136-stock, 2004-verified) for the GFC window.
Uses the full _FALLBACK_UNIVERSE for 2015-2026 and 2021-2026 windows.
"""
from __future__ import annotations

import datetime
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import logging

from src.utils.logging_config import setup_logging
from backtest.bt_inv_vol import run_backtest
from backtest._recession import DEFENSIVE_UNIVERSE, build_recession_months
from scripts.stress_test_2008 import UNIVERSE_2007

setup_logging("WARNING")

# Definitive S13 base config (without recession mode)
BASE = dict(
    top_n=5,
    max_weight=0.25,
    abs_momentum_filter=False,
    keep_n=8,
    weight_smoothing=0.5,
    sleeve_pct=0.10,
    sleeve_assets=("GLD",),
    sector_cap=0.80,
    cat_stop_pct=0.25,
)

CONFIGS = [
    ("S13 Definitive (no recession mode)",        {}),
    ("+ Recession → Defensives (all 24)",         dict(recession_mode=True, recession_defensives=DEFENSIVE_UNIVERSE)),
    ("+ Recession → Staples only (8 stocks)",     dict(recession_mode=True, recession_defensives=(
        "KO", "PEP", "PG", "CL", "MO", "MDLZ", "COST", "WMT",
    ))),
]

WINDOWS = [
    ("2007–2026 (GFC arc)",   datetime.date(2007, 1, 1), datetime.date(2026, 6, 1), UNIVERSE_2007),
    ("2015–2026 (full)",      datetime.date(2015, 1, 1), datetime.date(2026, 6, 1), None),
    ("2021–2026 (OOS)",       datetime.date(2021, 1, 1), datetime.date(2026, 6, 1), None),
]

COL = 40


def _cagr(ret_pct: float | None, start: datetime.date, end: datetime.date) -> float | None:
    if ret_pct is None:
        return None
    years = (end - start).days / 365.25
    return ((1 + ret_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else None


def _fmt(v: float | None, decimals: int = 2) -> str:
    return "   n/a" if v is None else f"{v:7.{decimals}f}"


def _print_recession_periods(start: datetime.date, end: datetime.date) -> None:
    months = build_recession_months(start - datetime.timedelta(days=400), end)
    if not months:
        print("  No recession periods in this window.")
        return
    sorted_months = sorted(months)
    # Find contiguous runs
    runs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    run_start = sorted_months[0]
    prev = sorted_months[0]
    for ym in sorted_months[1:]:
        # Check if consecutive month
        d_prev = datetime.date(prev[0], prev[1], 1)
        d_cur = datetime.date(ym[0], ym[1], 1)
        if (d_cur - d_prev).days > 35:
            runs.append((run_start, prev))
            run_start = ym
        prev = ym
    runs.append((run_start, prev))
    for r_start, r_end in runs:
        print(f"  Recession mode: {r_start[0]}-{r_start[1]:02d} → {r_end[0]}-{r_end[1]:02d}")


def run_window(
    label: str,
    start: datetime.date,
    end: datetime.date,
    universe: list[str] | None,
) -> None:
    w = 4 + COL + 8 + 8 + 9 + 8 + 9
    print(f"\n{'=' * w}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'=' * w}", flush=True)
    _print_recession_periods(start, end)
    print(f"\n  {'Config':<{COL}}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Stops':>8}{'SPY%':>9}", flush=True)
    print(f"  {'-' * (w - 4)}", flush=True)

    for cfg_label, extra in CONFIGS:
        kwargs = {**BASE, **extra}
        if universe is not None:
            kwargs["stock_universe"] = universe
        r = run_backtest(start_date=start, end_date=end, vol_window=20,
                         rebalance_freq="monthly", regime_filter=False,
                         sharpe_ranking=False, **kwargs)
        s, b = r["strategy"], r["benchmark"]
        cagr = _cagr(s.get("total_return_pct"), start, end)
        stops = s.get("cat_stop_triggers", 0)
        print(
            f"  {cfg_label:<{COL}}"
            f"{_fmt(cagr):>8}"
            f"{_fmt(s.get('sharpe_ratio')):>8}"
            f"{_fmt(s.get('max_drawdown_pct')):>9}"
            f"{stops:>8}"
            f"{_fmt(b.get('total_return_pct')):>9}",
            flush=True,
        )


def main() -> None:
    print(f"\nDefensive universe: {len(DEFENSIVE_UNIVERSE)} stocks")
    print("Recession signal: 2 consecutive negative GDP quarters → enter; 1 positive → exit (1-month pub lag)")
    for label, start, end, universe in WINDOWS:
        run_window(label, start, end, universe)
    print("\n⚠  Survivorship bias: universe excludes pre-2026 failures (Lehman etc.).")
    print("   2022 'recession' is technical (2 neg GDP quarters) — not NBER-declared.")
    print("   2025+ GDP values are placeholders; update _recession.py when BEA data is available.")


if __name__ == "__main__":
    main()
