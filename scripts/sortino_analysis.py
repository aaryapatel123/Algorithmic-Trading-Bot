"""Sortino analysis — evaluates all key strategy variants on downside-risk-adjusted return.

Sortino ratio = (CAGR − RF) / Downside Deviation (annualised, MAR = 0 %/month).
Unlike Sharpe, this penalises only negative months; upside volatility is not counted
against the strategy.

Primary question from Session 7 open work #1:
  Does keep8 + smooth 0.5 (CHALLENGER) beat cap25 + smooth 0.7 (PICK) on Sortino,
  or does the pick's higher Sharpe (0.823 vs 0.808) reflect genuinely better
  downside risk control?

Hypothesis: the challenger's lower Sharpe comes from larger upside months (more
upside variance), not from more frequent or deeper losses — Sortino should confirm
this by favouring the challenger.
"""
from __future__ import annotations

import datetime
import math
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest.bt_inv_vol import run_backtest
from backtest._price_cache import get_prices


RF_ANNUAL   = 0.04   # risk-free rate (matches Sharpe computation in run_backtest)
MAR_MONTHLY = 0.0    # minimum acceptable return per month; 0 = penalise every loss


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def sortino_metrics(monthly_rets_pct: list[float], cagr: float) -> dict:
    """Compute downside-risk metrics from a monthly-return series.

    Downside deviation uses all N observations in the denominator (not just
    negative months) — the standard academic definition, which gives a more
    conservative (lower) DD than dividing by the count of losing months only.
    """
    n = len(monthly_rets_pct)
    if n == 0:
        return {}

    # Downside deviation: sqrt(E[min(r − MAR, 0)^2]) × sqrt(12)
    dd_sq = [min(r / 100 - MAR_MONTHLY, 0) ** 2 for r in monthly_rets_pct]
    downside_dev_annual = math.sqrt(sum(dd_sq) / n * 12)

    sortino = (
        (cagr / 100 - RF_ANNUAL) / downside_dev_annual
        if downside_dev_annual > 0 else float("inf")
    )

    # Max drawdown from the monthly equity curve
    equity = [1.0]
    for r in monthly_rets_pct:
        equity.append(equity[-1] * (1 + r / 100))
    peak = 1.0
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak)

    # Gain-to-pain ratio: sum of positive months / |sum of negative months|
    pos = sum(r for r in monthly_rets_pct if r > 0)
    neg = abs(sum(r for r in monthly_rets_pct if r < 0))
    gain_pain = pos / neg if neg > 0 else float("inf")

    neg_months = sum(1 for r in monthly_rets_pct if r < 0)

    return {
        "sortino":      round(sortino, 3),
        "downside_dev": round(downside_dev_annual * 100, 2),   # as %
        "max_dd":       round(max_dd * 100, 2),                # as %
        "calmar":       round((cagr / 100) / max_dd, 3) if max_dd > 0 else float("inf"),
        "gain_pain":    round(gain_pain, 3),
        "neg_months":   neg_months,
        "pct_neg":      round(neg_months / n * 100, 1),
    }


def neg_months_by_year(monthly_with_dates: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for r in monthly_with_dates:
        yr = r["year"]
        counts.setdefault(yr, 0)
        if r["return_pct"] < 0:
            counts[yr] += 1
    return counts


def worst_months(monthly_with_dates: list[dict], n: int = 10) -> list[dict]:
    return sorted(monthly_with_dates, key=lambda x: x["return_pct"])[:n]


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def run_strategy(
    params: dict,
    start: datetime.date,
    end: datetime.date,
    cash: float,
) -> dict:
    """Wrap run_backtest and extract all metrics needed for Sortino analysis."""
    kwargs = dict(
        start_date=start,
        end_date=end,
        initial_cash=cash,
        abs_momentum_filter=False,
        regime_filter=False,
    )
    kwargs.update(params)
    result = run_backtest(**kwargs)

    total_ret_pct = result["strategy"]["total_return_pct"]
    years = (end - start).days / 365.25
    cagr = ((1 + total_ret_pct / 100) ** (1 / years) - 1) * 100

    monthly_with_dates = result["strategy"]["monthly_returns"]
    monthly_rets = [r["return_pct"] for r in monthly_with_dates]
    sharpe = result["strategy"]["sharpe_ratio"]

    metrics = sortino_metrics(monthly_rets, cagr)
    return {
        "cagr":               round(cagr, 1),
        "total_return":       round(total_ret_pct, 1),
        "sharpe":             round(sharpe, 3) if sharpe is not None else None,
        "trades":             result["strategy"]["total_trades"],
        "monthly_rets":       monthly_rets,
        "monthly_with_dates": monthly_with_dates,
        **metrics,
    }


# ---------------------------------------------------------------------------
# SPY reference
# ---------------------------------------------------------------------------

def spy_sortino_metrics(start: datetime.date, end: datetime.date) -> dict:
    df = get_prices("SPY", start - datetime.timedelta(days=45), end, auto_adjust=True)
    df.index = pd.to_datetime(df.index)

    # Monthly returns
    monthly_close = df["close"].resample("ME").last()
    monthly_pct = monthly_close.pct_change().dropna()
    monthly_rets = [
        r * 100
        for dt, r in monthly_pct.items()
        if start <= dt.date() <= end
    ]
    monthly_with_dates = [
        {"year": dt.year, "month": dt.month, "return_pct": round(r * 100, 4)}
        for dt, r in monthly_pct.items()
        if start <= dt.date() <= end
    ]

    # CAGR from in-range prices
    in_range = df["close"][df.index.date >= start]
    total_ret = float(in_range.iloc[-1] / in_range.iloc[0] - 1) * 100
    years = (end - start).days / 365.25
    cagr = ((1 + total_ret / 100) ** (1 / years) - 1) * 100

    # Daily Sharpe (matches _benchmark_metrics in bt_inv_vol)
    daily_rets = in_range.pct_change().dropna()
    rf_daily = RF_ANNUAL / 252
    excess = daily_rets - rf_daily
    sharpe = float(excess.mean() / excess.std() * (252 ** 0.5)) if excess.std() > 0 else None

    metrics = sortino_metrics(monthly_rets, cagr)
    return {
        "cagr":               round(cagr, 1),
        "total_return":       round(total_ret, 1),
        "sharpe":             round(sharpe, 3) if sharpe is not None else None,
        "trades":             0,
        "monthly_rets":       monthly_rets,
        "monthly_with_dates": monthly_with_dates,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    START = datetime.date(2015, 1, 1)
    END   = datetime.date(2026, 1, 1)
    CASH  = 100_000.0

    strategies = [
        ("Base no-cap",                  dict(top_n=5, max_weight=1.0)),
        ("Base cap25",                   dict(top_n=5, max_weight=0.25)),
        ("cap25 + smooth 0.7 (PICK)",    dict(top_n=5, max_weight=0.25, weight_smoothing=0.7)),
        ("cap25 + smooth 0.5",           dict(top_n=5, max_weight=0.25, weight_smoothing=0.5)),
        ("keep8 + cap25",                dict(top_n=5, max_weight=0.25, keep_n=8)),
        ("keep8 + smooth 0.5 (CHALL.)",  dict(top_n=5, max_weight=0.25, keep_n=8, weight_smoothing=0.5)),
        ("Vol-target 15% + cap25",       dict(top_n=5, max_weight=0.25, vol_targeting=True, vol_target=0.15)),
    ]

    print("Running backtests...", flush=True)
    results: dict[str, dict] = {}
    for name, params in strategies:
        print(f"  {name}", flush=True)
        results[name] = run_strategy(params, START, END, CASH)

    print("  SPY buy-and-hold", flush=True)
    results["SPY buy-and-hold"] = spy_sortino_metrics(START, END)

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    print("=" * 115)
    print(f"SORTINO ANALYSIS — All Strategies (2015–2026, RF={RF_ANNUAL*100:.0f}%, MAR=0%/month)")
    print("=" * 115)
    col = 32
    header = (
        f"{'Strategy':<{col}} | {'CAGR':>6} | {'Sortino':>8} | "
        f"{'DwnDev':>7} | {'MaxDD':>6} | {'Calmar':>7} | "
        f"{'Sharpe':>7} | {'G/Pain':>7} | {'Neg%':>5} | {'Trades':>7}"
    )
    print(header)
    print("-" * 115)

    for name, r in results.items():
        sharpe_str = f"{r['sharpe']:.3f}" if r["sharpe"] is not None else "  N/A"
        print(
            f"{name:<{col}} | {r['cagr']:>5.1f}% | {r['sortino']:>8.3f} | "
            f"{r['downside_dev']:>6.2f}% | {r['max_dd']:>5.1f}% | "
            f"{r['calmar']:>7.3f} | {sharpe_str:>7} | "
            f"{r['gain_pain']:>7.3f} | {r['pct_neg']:>4.1f}% | "
            f"{r['trades']:>7}"
        )

    # ── PICK vs CHALLENGER: head-to-head ─────────────────────────────────────
    pick  = results["cap25 + smooth 0.7 (PICK)"]
    chall = results["keep8 + smooth 0.5 (CHALL.)"]

    print()
    print("=" * 80)
    print("PICK vs CHALLENGER — head-to-head comparison")
    print("=" * 80)
    print(f"{'Metric':<28} | {'cap25+smooth0.7 (PICK)':>22} | {'keep8+smooth0.5 (CHALL)':>23} | {'Winner':>10}")
    print("-" * 80)

    comparisons = [
        ("CAGR (%)",              pick["cagr"],         chall["cagr"],         "higher"),
        ("Sortino ratio",         pick["sortino"],       chall["sortino"],      "higher"),
        ("Downside dev (%)",      pick["downside_dev"],  chall["downside_dev"], "lower"),
        ("Max DD (%)",            pick["max_dd"],        chall["max_dd"],       "lower"),
        ("Calmar ratio",          pick["calmar"],        chall["calmar"],       "higher"),
        ("Sharpe ratio",          pick["sharpe"],        chall["sharpe"],       "higher"),
        ("Gain-to-Pain ratio",    pick["gain_pain"],     chall["gain_pain"],    "higher"),
        ("Negative months (%)",   pick["pct_neg"],       chall["pct_neg"],      "lower"),
        ("Negative month count",  pick["neg_months"],    chall["neg_months"],   "lower"),
        ("Trade count",           pick["trades"],        chall["trades"],       "lower"),
    ]

    pick_wins = 0
    chall_wins = 0
    for label, p_val, c_val, better in comparisons:
        if better == "higher":
            winner = "PICK" if (p_val or 0) > (c_val or 0) else "CHALLENGER"
        else:
            winner = "PICK" if (p_val or 0) < (c_val or 0) else "CHALLENGER"
        if winner == "PICK":
            pick_wins += 1
        else:
            chall_wins += 1
        p_str = f"{p_val}" if isinstance(p_val, int) else f"{p_val}"
        c_str = f"{c_val}" if isinstance(c_val, int) else f"{c_val}"
        print(f"{label:<28} | {p_str:>22} | {c_str:>23} | {winner:>10}")

    print("-" * 80)
    print(f"{'Scorecard':<28} | {'PICK wins: ' + str(pick_wins):>22} | {'CHALL wins: ' + str(chall_wins):>23} |")

    # ── Variance decomposition: upside vs downside ────────────────────────────
    print()
    print("── Variance decomposition: upside vs downside volatility ──")
    print("(shows whether the Sharpe gap is from more upside or more downside variance)")
    print()

    def variance_decomp(name: str, monthly_rets: list[float], cagr: float) -> None:
        n = len(monthly_rets)
        pos_sq = [max(r / 100, 0) ** 2 for r in monthly_rets]
        neg_sq = [min(r / 100, 0) ** 2 for r in monthly_rets]
        upside_dev  = math.sqrt(sum(pos_sq) / n * 12) * 100
        downside_dev = math.sqrt(sum(neg_sq) / n * 12) * 100
        total_dev   = math.sqrt((sum(pos_sq) + sum(neg_sq)) / n * 12) * 100
        upside_pct  = sum(pos_sq) / (sum(pos_sq) + sum(neg_sq)) * 100 if (sum(pos_sq) + sum(neg_sq)) > 0 else 0

        print(f"  {name}")
        print(f"    Total annualised vol : {total_dev:.2f}%")
        print(f"    Upside deviation     : {upside_dev:.2f}%  ({upside_pct:.0f}% of total variance)")
        print(f"    Downside deviation   : {downside_dev:.2f}%  ({100-upside_pct:.0f}% of total variance)")
        print()

    variance_decomp("cap25 + smooth 0.7 (PICK)",   pick["monthly_rets"],  pick["cagr"])
    variance_decomp("keep8 + smooth 0.5 (CHALL.)", chall["monthly_rets"], chall["cagr"])

    # ── Year-by-year negative months ─────────────────────────────────────────
    print("── Negative months by year ──")
    spy_r = results["SPY buy-and-hold"]
    pick_by_yr  = neg_months_by_year(pick["monthly_with_dates"])
    chall_by_yr = neg_months_by_year(chall["monthly_with_dates"])
    spy_by_yr   = neg_months_by_year(spy_r["monthly_with_dates"])

    all_years = sorted(set(pick_by_yr) | set(chall_by_yr) | set(spy_by_yr))
    print(f"{'Year':>6} | {'PICK (neg/12)':>14} | {'CHALL (neg/12)':>15} | {'SPY (neg/12)':>13}")
    print("-" * 57)
    for yr in all_years:
        p  = pick_by_yr.get(yr, 0)
        c  = chall_by_yr.get(yr, 0)
        s  = spy_by_yr.get(yr, 0)
        flag = "  ← PICK worse" if p > c else ("  ← CHALL worse" if c > p else "")
        print(f"{yr:>6} | {p:>14} | {c:>15} | {s:>13}{flag}")

    # ── Worst 10 months for each ──────────────────────────────────────────────
    print()
    print("── 10 worst months: PICK vs CHALLENGER ──")
    pick_worst  = worst_months(pick["monthly_with_dates"],  10)
    chall_worst = worst_months(chall["monthly_with_dates"], 10)

    print(f"  {'PICK (cap25+smooth0.7)':<30}  |  {'CHALLENGER (keep8+smooth0.5)':<30}")
    print(f"  {'Date':<12} {'Return':>8}           |  {'Date':<12} {'Return':>8}")
    print(f"  {'-'*30}     |  {'-'*30}")
    for pw, cw in zip(pick_worst, chall_worst):
        pd_str = f"{pw['year']}-{pw['month']:02d}"
        cd_str = f"{cw['year']}-{cw['month']:02d}"
        print(f"  {pd_str:<12} {pw['return_pct']:>7.2f}%         |  {cd_str:<12} {cw['return_pct']:>7.2f}%")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    if chall["sortino"] > pick["sortino"] and chall["calmar"] > pick["calmar"]:
        print("  CHALLENGER wins on both Sortino and Calmar.")
        print("  The lower Sharpe reflects more upside variance, not worse downside risk.")
        print("  >> Promote keep8 + smooth 0.5 + cap25 as the new definitive pick.")
    elif pick["sortino"] > chall["sortino"] and pick["calmar"] > chall["calmar"]:
        print("  PICK wins on both Sortino and Calmar.")
        print("  >> Retain cap25 + smooth 0.7 as the definitive pick.")
    else:
        p_sor, c_sor = pick["sortino"], chall["sortino"]
        p_cal, c_cal = pick["calmar"],  chall["calmar"]
        print(f"  Mixed result: Sortino → {'PICK' if p_sor > c_sor else 'CHALLENGER'} "
              f"({p_sor:.3f} vs {c_sor:.3f}), "
              f"Calmar → {'PICK' if p_cal > c_cal else 'CHALLENGER'} "
              f"({p_cal:.3f} vs {c_cal:.3f}).")
        print("  No clear winner — further analysis or additional evidence needed.")
    print()


if __name__ == "__main__":
    main()
