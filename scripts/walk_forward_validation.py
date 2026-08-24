#!/usr/bin/env python3
"""Walk-forward validation — production config, IS vs OOS split.

Train window : 2015-01-01 → 2020-12-31  (in-sample,  6 years)
Test  window : 2021-01-01 → 2026-01-01  (out-of-sample, 5 years)
Full  window : 2015-01-01 → 2026-01-01  (reference)

Key question: does Sortino ≥ 1.8 and CAGR ≥ 20% survive cold on OOS data?
Degradation ratio (OOS/IS) < 0.6 on Sortino = likely overfit.

    python scripts/walk_forward_validation.py
"""
from __future__ import annotations

import datetime
import math
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest.bt_inv_vol import run_backtest

RF_ANNUAL = 0.04

IS_START  = datetime.date(2015, 1, 1)
IS_END    = datetime.date(2020, 12, 31)
OOS_START = datetime.date(2021, 1, 1)
OOS_END   = datetime.date(2026, 1, 1)
FULL_END  = datetime.date(2026, 1, 1)

PROD = dict(
    initial_cash=100_000.0,
    top_n=5,
    keep_n=8,
    max_weight=0.25,
    weight_smoothing=0.5,
    vol_window=20,
    abs_momentum_filter=False,
    regime_filter=False,
)


def metrics(res: dict, start: datetime.date, end: datetime.date) -> dict:
    tr   = res["strategy"]["total_return_pct"]
    yrs  = (end - start).days / 365.25
    cagr = ((1 + tr / 100) ** (1 / yrs) - 1) * 100

    monthly = res["strategy"]["monthly_returns"]
    mr = [r["return_pct"] for r in monthly]
    n  = len(mr)

    dd_sq   = [min(r / 100, 0) ** 2 for r in mr]
    ddev    = math.sqrt(sum(dd_sq) / n * 12) if n else 0.0
    sortino = (cagr / 100 - RF_ANNUAL) / ddev if ddev > 0 else float("inf")

    eq = [1.0]
    for r in mr:
        eq.append(eq[-1] * (1 + r / 100))
    peak, mdd = 1.0, 0.0
    for v in eq:
        peak = max(peak, v)
        mdd  = max(mdd, (peak - v) / peak)
    calmar = (cagr / 100) / mdd if mdd > 0 else float("inf")

    by_year: dict[int, float] = defaultdict(lambda: 1.0)
    for r in monthly:
        by_year[r["year"]] *= (1 + r["return_pct"] / 100)
    annual = {y: (v - 1) * 100 for y, v in sorted(by_year.items())}

    return {
        "cagr":    cagr,
        "sortino": sortino,
        "calmar":  calmar,
        "maxdd":   mdd * 100,
        "trades":  res["strategy"]["total_trades"],
        "annual":  annual,
    }


def _bar(label: str, width: int = 26) -> str:
    return f"{label:<{width}}"


def spy_cagr(res: dict, start: datetime.date, end: datetime.date) -> tuple[float, float, float]:
    """Return (cagr, sharpe, maxdd) for the SPY benchmark embedded in a result dict."""
    b   = res.get("benchmark", {})
    tr  = b.get("total_return_pct", float("nan"))
    yrs = (end - start).days / 365.25
    cagr = ((1 + tr / 100) ** (1 / yrs) - 1) * 100 if not math.isnan(tr) else float("nan")
    return cagr, b.get("sharpe_ratio", float("nan")), b.get("max_drawdown_pct", float("nan"))


def main() -> None:
    windows = [
        ("In-sample  (2015–2020)", IS_START,  IS_END),
        ("OOS        (2021–2025)", OOS_START, OOS_END),
        ("Full       (2015–2025)", IS_START,  FULL_END),
    ]

    results = []
    for label, start, end in windows:
        print(f"  running {label} ...")
        res  = run_backtest(start_date=start, end_date=end, **PROD)
        m    = metrics(res, start, end)
        scagr, ssharpe, smaxdd = spy_cagr(res, start, end)
        results.append((label, start, end, m, scagr, ssharpe, smaxdd))

    is_m, oos_m = results[0][3], results[1][3]
    degrad_sortino = oos_m["sortino"] / is_m["sortino"] if is_m["sortino"] > 0 else float("nan")
    degrad_cagr    = oos_m["cagr"]    / is_m["cagr"]    if is_m["cagr"]    > 0 else float("nan")

    # ── Core metrics table ──────────────────────────────────────────────────
    hdr = (
        f"\n{'Window':<26}{'CAGR':>7}{'Sortino':>9}{'Calmar':>8}"
        f"{'MaxDD':>8}  │  {'SPY CAGR':>9}{'SPY Sharpe':>11}{'SPY MaxDD':>10}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for label, start, end, m, scagr, ssharpe, smaxdd in results:
        edge = m["cagr"] - scagr
        print(
            f"{_bar(label)}{m['cagr']:>6.1f}%{m['sortino']:>9.3f}"
            f"{m['calmar']:>8.3f}{m['maxdd']:>7.1f}%"
            f"  │  {scagr:>8.1f}%{ssharpe:>11.3f}{smaxdd:>9.1f}%"
            f"   edge={edge:+.1f}pp"
        )
    print(sep)
    print(f"  Sortino degradation ratio  OOS/IS = {degrad_sortino:.2f}  (≥0.70 = healthy)")
    print(f"  CAGR    degradation ratio  OOS/IS = {degrad_cagr:.2f}  (≥0.70 = healthy)")

    # ── Annual returns: strategy vs SPY ────────────────────────────────────
    is_ann  = is_m["annual"]
    oos_ann = oos_m["annual"]
    all_ann = {**is_ann, **oos_ann}

    print(f"\n{'Year':<8}{'Strategy':>10}{'Window':>10}")
    print("-" * 30)
    for yr in sorted(all_ann):
        window = "IS" if yr in is_ann else "OOS"
        print(f"{yr:<8}{all_ann[yr]:>+9.1f}%{window:>10}")

    print(sep)
    if degrad_sortino >= 0.70:
        verdict = "PASS — strategy generalises. OOS Sortino within healthy range."
    elif degrad_sortino >= 0.50:
        verdict = "MARGINAL — moderate degradation. Proceed with caution; recheck params."
    else:
        verdict = "FAIL — heavy degradation. Params likely overfit to 2015-2020."
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
