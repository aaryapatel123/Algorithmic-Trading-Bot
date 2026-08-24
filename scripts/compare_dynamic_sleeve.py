#!/usr/bin/env python3
"""Dynamic GLD sleeve sweep.

Tests whether expanding the GLD sleeve from 10% → 20% during market stress
improves drawdown without sacrificing too much CAGR.

Stress triggers tested:
  A) SPY 21-day return < -8% (momentum crash signal)
  B) VIX > 30 (fear spike signal)
  C) Either A or B
  D) Either, expand to 25% (larger buffer)

Baseline: S13 definitive (static 10% GLD, sector_cap=80%, cat_stop=25%).
"""
from __future__ import annotations

import datetime
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import logging
from backtest.bt_inv_vol import run_backtest
from scripts.stress_test_2008 import UNIVERSE_2007

logging.getLogger().setLevel(logging.WARNING)

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

CONFIGS: list[tuple[str, dict]] = [
    ("Static 10% GLD (S13 baseline)",
     {}),
    ("Dynamic 10%→20%: SPY<-8% only",
     dict(dynamic_sleeve=True, sleeve_pct_stress=0.20, vix_threshold=0, spy_monthly_dd=-0.08)),
    ("Dynamic 10%→20%: VIX>30 only",
     dict(dynamic_sleeve=True, sleeve_pct_stress=0.20, vix_threshold=30.0, spy_monthly_dd=-1.0)),
    ("Dynamic 10%→20%: either trigger",
     dict(dynamic_sleeve=True, sleeve_pct_stress=0.20, vix_threshold=30.0, spy_monthly_dd=-0.08)),
    ("Dynamic 10%→25%: either trigger",
     dict(dynamic_sleeve=True, sleeve_pct_stress=0.25, vix_threshold=30.0, spy_monthly_dd=-0.08)),
]

WINDOWS = [
    ("2007–2026 (GFC)", datetime.date(2007, 1, 1), datetime.date(2026, 6, 1), UNIVERSE_2007),
    ("2015–2026 (full)", datetime.date(2015, 1, 1), datetime.date(2026, 6, 1), None),
    ("2021–2026 (OOS)",  datetime.date(2021, 1, 1), datetime.date(2026, 6, 1), None),
]

COL = 42


def _cagr(ret: float | None, start: datetime.date, end: datetime.date) -> float | None:
    if ret is None:
        return None
    yrs = (end - start).days / 365.25
    return ((1 + ret / 100) ** (1 / yrs) - 1) * 100 if yrs > 0 else None


def _f(v: float | None) -> str:
    return "   n/a" if v is None else f"{v:7.2f}"


def run_window(label: str, start: datetime.date, end: datetime.date, universe) -> None:
    w = 4 + COL + 8 + 8 + 9 + 8
    print(f"\n{'=' * w}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'=' * w}", flush=True)
    print(f"  {'Config':<{COL}}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Stops':>8}", flush=True)
    print(f"  {'-' * (w - 4)}", flush=True)

    for cfg_label, extra in CONFIGS:
        kw = {**BASE, **extra}
        if universe is not None:
            kw["stock_universe"] = universe
        r = run_backtest(start_date=start, end_date=end, vol_window=20,
                         rebalance_freq="monthly", regime_filter=False,
                         sharpe_ranking=False, **kw)
        s = r["strategy"]
        cagr = _cagr(s.get("total_return_pct"), start, end)
        stops = s.get("cat_stop_triggers", 0)
        print(
            f"  {cfg_label:<{COL}}"
            f"{_f(cagr):>8}"
            f"{_f(s.get('sharpe_ratio')):>8}"
            f"{_f(s.get('max_drawdown_pct')):>9}"
            f"{stops:>8}",
            flush=True,
        )


def main() -> None:
    print("\nDynamic GLD sleeve sweep — 10% base, 20%/25% during stress")
    print("Stress triggers: SPY 21-day return < -8%  |  VIX daily close > 30")
    for label, start, end, universe in WINDOWS:
        run_window(label, start, end, universe)
    print("\n⚠  GFC window uses UNIVERSE_2007 (136 stocks, survivorship-biased).")
    print("   VIX trigger requires live ^VIX fetch (not in local pickle cache).")


if __name__ == "__main__":
    main()
