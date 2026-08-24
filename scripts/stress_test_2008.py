#!/usr/bin/env python3
"""2008 / GFC stress test for the momentum + inv-vol strategy.

Runs the primary strategy (top-5, inv-vol, 25% cap, monthly, abs OFF) and the
base no-cap variant on a 2004-verified 136-stock universe over three windows:
  • 2007-2010  — the crash + trough + initial recovery
  • 2007-2015  — full GFC + recovery arc through to the main backtest start
  • 2015-2026  — same universe, main window (sanity: should reproduce ~0.90 Sharpe)

The 136-stock universe excludes all names that lacked data before 2004 (PYPL,
KHC, TSLA, META, V, MA, ABBV, ANET, AVGO, ICE, GM, NOW, PM, PSX, TDG, TMUS,
ZTS, WBA, MMC, GOOG/GOOGL with <200 bars, CRM short). This fixes the
backtrader sync-freeze that made any pre-2015 run produce 0% returns.

IMPORTANT CAVEAT (printed at end of each run):
  This universe is today's large-caps -- it suffers from SURVIVORSHIP BIAS.
  Any stock that went bankrupt or was acquired 2004-2015 (e.g. Lehman, Bear
  Stearns, Wachovia, Washington Mutual) is NOT in this list. The true 2007
  momentum universe would have included those names. Results here are an
  OPTIMISTIC lower bound on how badly 2008 could hurt.
"""
from __future__ import annotations

import datetime
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.utils.logging_config import setup_logging
from backtest.bt_inv_vol import run_backtest

setup_logging("WARNING")
logger = logging.getLogger(__name__)

# 2004-verified subset of _FALLBACK_UNIVERSE (136 stocks).
UNIVERSE_2007 = [
    "AAPL","ABT","ACN","ADBE","ADI","ADP","ADSK","AEP","AFL","AIG",
    "AMAT","AMD","AMGN","AMZN","AXP","BA","BAC","BDX","BIIB","BK",
    "BKNG","BLK","BMY","BSX","C","CAT","CB","CCI","CDNS","CI",
    "CL","CMCSA","CME","COF","COP","COST","CSCO","CTAS","CVS","CVX",
    "D","DD","DE","DHR","DIS","DUK","ECL","EL","EMR","EOG",
    "EW","EXC","F","FDX","FIS","FISV","GD","GE","GILD","GPN",
    "GS","HD","HON","IBM","INTC","INTU","ISRG","ITW","JNJ","JPM",
    "KLAC","KO","LIN","LLY","LMT","LOW","LRCX","MCD","MCHP","MCK",
    "MDLZ","MDT","MET","MMM","MO","MRK","MS","MSFT","MU","NEE",
    "NFLX","NKE","NOC","NSC","NVDA","ORCL","OXY","PEP","PFE","PG",
    "PGR","PLD","PNC","PSA","QCOM","REGN","ROP","RTX","SBUX","SCHW",
    "SHW","SLB","SNPS","SO","SPG","SPGI","SYK","T","TFC","TGT",
    "TJX","TMO","TT","TXN","UNH","UNP","UPS","USB","VLO","VRTX",
    "VZ","WFC","WM","WMT","XEL","XOM",
]

SURVIVORSHIP_WARNING = """
  *** SURVIVORSHIP BIAS WARNING ***
  This universe contains only stocks that survived to 2026 as large-caps.
  The real 2007 momentum top-5 could have included Lehman Brothers, Bear
  Stearns, Wachovia, Washington Mutual, Citigroup (near-zero), AIG (−97%),
  and others. Those names are absent here. These numbers are an OPTIMISTIC
  estimate of 2008 performance — real results would have been worse.
"""

PERIODS = [
    ("2007-2010  (crash + trough + initial recovery)", datetime.date(2007, 1, 1), datetime.date(2010, 12, 31)),
    ("2007-2015  (full GFC arc to main backtest start)", datetime.date(2007, 1, 1), datetime.date(2015, 1, 1)),
    ("2015-2026  (main window, sanity check vs ~0.90)", datetime.date(2015, 1, 1), datetime.date(2026, 6, 1)),
]

CONFIGS = [
    ("Primary (top5 + cap25)",          dict(top_n=5, max_weight=0.25, abs_momentum_filter=False)),
    ("cap25 + smooth 0.7 (old pick)",   dict(top_n=5, max_weight=0.25, abs_momentum_filter=False, weight_smoothing=0.7)),
    ("keep8 + smooth 0.5 (NEW PICK)",   dict(top_n=5, max_weight=0.25, abs_momentum_filter=False, keep_n=8, weight_smoothing=0.5)),
    ("Base no-cap (top5)",              dict(top_n=5, max_weight=1.0,  abs_momentum_filter=False)),
    # Session 12 definitive — GLD 10% sleeve, no stop
    ("S12: keep8+smooth+GLD10%",        dict(top_n=5, max_weight=0.25, abs_momentum_filter=False,
                                             keep_n=8, weight_smoothing=0.5,
                                             sleeve_pct=0.10, sleeve_assets=("GLD",))),
    # Session 13 definitive — GLD 10%, sector_cap 80%, catastrophic stop 25%
    ("★ S13 DEFINITIVE (stop25%)",      dict(top_n=5, max_weight=0.25, abs_momentum_filter=False,
                                             keep_n=8, weight_smoothing=0.5,
                                             sleeve_pct=0.10, sleeve_assets=("GLD",),
                                             sector_cap=0.80, cat_stop_pct=0.25)),
]

COL = 30


def _fmt(v):
    return "    n/a" if v is None else f"{v:7.2f}"


def _cagr(total_return_pct: float | None, start: datetime.date, end: datetime.date) -> float | None:
    if total_return_pct is None:
        return None
    years = (end - start).days / 365.25
    if years <= 0:
        return None
    return ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100


def _header(title: str, start: datetime.date, end: datetime.date) -> None:
    w = 4 + COL + 8 + 9 + 9 + 9 + 10 + 6
    print("\n" + "=" * w, flush=True)
    print(f"  {title}", flush=True)
    print("=" * w, flush=True)
    print(
        f"  {'Config':<{COL}}{'CAGR%':>8}{'Sharpe':>9}{'MaxDD%':>9}{'Stops':>9}{'SPY Ret%':>10}{'SPY Shp':>6}",
        flush=True,
    )
    print("  " + "-" * (w - 4), flush=True)


def run_period(label: str, start: datetime.date, end: datetime.date) -> None:
    _header(label, start, end)
    for cfg_label, kwargs in CONFIGS:
        logger.warning("Running %s | %s -> %s", cfg_label, start, end)
        r = run_backtest(
            start_date=start, end_date=end,
            stock_universe=UNIVERSE_2007,
            vol_window=20, rebalance_freq="monthly",
            regime_filter=False, sharpe_ranking=False,
            **kwargs,
        )
        s, b = r["strategy"], r["benchmark"]
        cagr = _cagr(s.get("total_return_pct"), start, end)
        stops = s.get("cat_stop_triggers", 0)
        print(
            f"  {cfg_label:<{COL}}"
            f"{_fmt(cagr):>8}"
            f"{_fmt(s.get('sharpe_ratio')):>9}"
            f"{_fmt(s.get('max_drawdown_pct')):>9}"
            f"{stops:>9}"
            f"{_fmt(b.get('total_return_pct')):>10}"
            f"{_fmt(b.get('sharpe_ratio')):>6}",
            flush=True,
        )
    print(SURVIVORSHIP_WARNING, flush=True)


def main() -> None:
    print(f"\nUniverse: {len(UNIVERSE_2007)} stocks (2004-verified, survivorship-biased)")
    for label, start, end in PERIODS:
        run_period(label, start, end)
    print("\nNote: CAGR computed from total return + window length. Sortino/Calmar require sortino_analysis.py.")


if __name__ == "__main__":
    main()
