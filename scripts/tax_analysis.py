"""After-tax CAGR analysis.

For each strategy, captures every individual trade (symbol, entry date, exit
date, P&L), classifies it as short-term (<365 days) or long-term (>=365 days),
applies federal capital-gains tax year-by-year, and reports after-tax CAGR
alongside the pre-tax figure.

Two tax brackets are run side-by-side:
  High  : ST 37% / LT 23.8%  (top federal bracket + 3.8% NIIT)
  Medium: ST 24% / LT 15%    (middle federal bracket)

State taxes are excluded — add your state rate to both ST and LT if needed.
"""
from __future__ import annotations

import datetime
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any

import backtrader as bt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest.bt_inv_vol import MomentumInvVolStrategy, _fetch_feed, _benchmark_metrics
from src.strategy.combined_momentum import _FALLBACK_UNIVERSE
from backtest._price_cache import get_prices


# ---------------------------------------------------------------------------
# Custom analyzer — captures individual closed trades
# ---------------------------------------------------------------------------

class TaxTradeAnalyzer(bt.Analyzer):
    """Records each closed trade with symbol, open/close date, and net P&L."""

    def start(self):
        self.trades: list[dict] = []

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        open_dt = bt.num2date(trade.dtopen).date()
        close_dt = bt.num2date(trade.dtclose).date()
        holding_days = (close_dt - open_dt).days
        self.trades.append({
            "symbol": trade.data._name,
            "open_date": open_dt,
            "close_date": close_dt,
            "holding_days": holding_days,
            "pnl": trade.pnlcomm,  # after commission
            "long_term": holding_days >= 365,
        })

    def get_analysis(self):
        return self.trades


# ---------------------------------------------------------------------------
# Tax computation
# ---------------------------------------------------------------------------

def apply_taxes(
    trades: list[dict],
    initial_cash: float,
    st_rate: float,
    lt_rate: float,
    start_date: datetime.date,
    end_date: datetime.date,
) -> dict:
    """
    Simulate year-by-year tax payments on realised gains/losses.

    Approach:
    - Gains are realised when a trade closes (close_date).
    - Tax is computed at year-end on net annual gains (losses carry forward
      within the same character — ST offsets ST, LT offsets LT).
    - Tax is deducted from the portfolio value at year-end.
    - Loss carryforward across years (simplified: ST losses carry as ST,
      LT losses carry as LT; no wash-sale adjustment).
    """
    # Bucket gains/losses by year and type
    by_year: dict[int, dict[str, float]] = defaultdict(lambda: {"st": 0.0, "lt": 0.0})
    for t in trades:
        if t["close_date"] < start_date or t["close_date"] > end_date:
            continue
        yr = t["close_date"].year
        if t["long_term"]:
            by_year[yr]["lt"] += t["pnl"]
        else:
            by_year[yr]["st"] += t["pnl"]

    # Carry-forward losses
    st_carry = 0.0
    lt_carry = 0.0

    total_tax_paid = 0.0
    year_rows = []

    for yr in sorted(by_year.keys()):
        st_gain = by_year[yr]["st"] + st_carry
        lt_gain = by_year[yr]["lt"] + lt_carry

        # Losses offset gains of same character first, then cross-character
        if st_gain < 0 and lt_gain > 0:
            offset = min(-st_gain, lt_gain)
            lt_gain -= offset
            st_gain += offset
        if lt_gain < 0 and st_gain > 0:
            offset = min(-lt_gain, st_gain)
            st_gain -= offset
            lt_gain += offset

        st_tax = max(st_gain, 0) * st_rate
        lt_tax = max(lt_gain, 0) * lt_rate
        tax = st_tax + lt_tax
        total_tax_paid += tax

        st_carry = min(st_gain, 0)
        lt_carry = min(lt_gain, 0)

        year_rows.append({
            "year": yr,
            "st_gain": round(st_gain, 0),
            "lt_gain": round(lt_gain, 0),
            "tax": round(tax, 0),
        })

    return {"total_tax": total_tax_paid, "by_year": year_rows,
            "st_carry": st_carry, "lt_carry": lt_carry}


# ---------------------------------------------------------------------------
# Modified run that returns trade-level data
# ---------------------------------------------------------------------------

def run_with_trades(
    top_n: int = 5,
    max_weight: float = 0.25,
    weight_smoothing: float = 1.0,
    keep_n: int = 0,
    abs_momentum_filter: bool = False,
    start_date: datetime.date = datetime.date(2015, 1, 1),
    end_date: datetime.date = datetime.date(2026, 1, 1),
    initial_cash: float = 100_000.0,
) -> dict:
    data_start = start_date - datetime.timedelta(days=380)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MomentumInvVolStrategy,
        top_n=top_n,
        max_weight=max_weight,
        weight_smoothing=weight_smoothing,
        keep_n=keep_n,
        abs_momentum_filter=abs_momentum_filter,
        rebalance_freq="monthly",
        regime_filter=False,
        printlog=False,
    )

    loaded = []
    for sym in ["SPY", "AGG"] + list(_FALLBACK_UNIVERSE):
        try:
            feed = _fetch_feed(sym, data_start, end_date)
            cerebro.adddata(feed, name=sym)
            if sym not in ("SPY", "AGG"):
                loaded.append(sym)
        except Exception as exc:
            pass

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(TaxTradeAnalyzer, _name="tax_trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="monthly_returns",
                        timeframe=bt.TimeFrame.Months)

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    years = (end_date - start_date).days / 365.25
    pretax_cagr = ((1 + total_return / 100) ** (1 / years) - 1) * 100

    trades = strat.analyzers.tax_trades.get_analysis()

    return {
        "final_value": final_value,
        "total_return_pct": total_return,
        "pretax_cagr": pretax_cagr,
        "trades": trades,
        "initial_cash": initial_cash,
        "years": years,
    }


# ---------------------------------------------------------------------------
# After-tax CAGR: deduct annual taxes from portfolio value
# ---------------------------------------------------------------------------

def after_tax_cagr(
    pretax_final: float,
    initial_cash: float,
    tax_result: dict,
    years: float,
) -> float:
    after_tax_final = pretax_final - tax_result["total_tax"]
    if after_tax_final <= 0:
        return float("-inf")
    return ((after_tax_final / initial_cash) ** (1 / years) - 1) * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    START = datetime.date(2015, 1, 1)
    END   = datetime.date(2026, 1, 1)
    CASH  = 100_000.0

    strategies = [
        ("Base cap25",                dict(top_n=5, max_weight=0.25)),
        ("cap25 + smooth 0.7 (PICK)", dict(top_n=5, max_weight=0.25, weight_smoothing=0.7)),
        ("cap25 + smooth 0.5",        dict(top_n=5, max_weight=0.25, weight_smoothing=0.5)),
        ("keep8 + cap25",             dict(top_n=5, max_weight=0.25, keep_n=8)),
        ("keep8 + smooth 0.5 + cap25",dict(top_n=5, max_weight=0.25, keep_n=8, weight_smoothing=0.5)),
    ]

    BRACKETS = [
        ("High  (ST 37% / LT 23.8%)", 0.37, 0.238),
        ("Medium(ST 24% / LT 15%  )", 0.24, 0.15),
    ]

    # Header
    col = 30
    print(f"\n{'Strategy':<{col}} | {'Pre-tax CAGR':>13} | {'Trades':>7} | {'% LT':>6} | "
          f"{'AT CAGR (High)':>14} | {'AT CAGR (Med)':>13} | {'Tax paid (High)':>15}")
    print("-" * 115)

    for name, params in strategies:
        r = run_with_trades(start_date=START, end_date=END, initial_cash=CASH, **params)
        trades = r["trades"]
        n_trades = len(trades)
        pct_lt = (sum(1 for t in trades if t["long_term"]) / n_trades * 100) if n_trades else 0

        at_cagrs = []
        taxes = []
        for label, st_rate, lt_rate in BRACKETS:
            tx = apply_taxes(trades, CASH, st_rate, lt_rate, START, END)
            atc = after_tax_cagr(r["final_value"], CASH, tx, r["years"])
            at_cagrs.append(atc)
            taxes.append(tx["total_tax"])

        print(f"{name:<{col}} | {r['pretax_cagr']:>12.1f}% | {n_trades:>7} | {pct_lt:>5.0f}% | "
              f"{at_cagrs[0]:>13.1f}% | {at_cagrs[1]:>12.1f}% | ${taxes[0]:>14,.0f}")

    # Separator then detail breakdown for the two main picks
    print()
    print("── Year-by-year tax detail: cap25 + smooth 0.7 (High bracket) ──")
    r = run_with_trades(start_date=START, end_date=END, initial_cash=CASH,
                        top_n=5, max_weight=0.25, weight_smoothing=0.7)
    tx = apply_taxes(r["trades"], CASH, 0.37, 0.238, START, END)
    print(f"{'Year':>6} | {'ST gain':>10} | {'LT gain':>10} | {'Tax':>10}")
    print("-" * 45)
    for row in tx["by_year"]:
        print(f"{row['year']:>6} | ${row['st_gain']:>9,.0f} | ${row['lt_gain']:>9,.0f} | ${row['tax']:>9,.0f}")
    print(f"{'TOTAL':>6} |             |             | ${tx['total_tax']:>9,.0f}")

    print()
    print("── Year-by-year tax detail: keep8 + smooth 0.5 + cap25 (High bracket) ──")
    r2 = run_with_trades(start_date=START, end_date=END, initial_cash=CASH,
                         top_n=5, max_weight=0.25, keep_n=8, weight_smoothing=0.5)
    tx2 = apply_taxes(r2["trades"], CASH, 0.37, 0.238, START, END)
    print(f"{'Year':>6} | {'ST gain':>10} | {'LT gain':>10} | {'Tax':>10}")
    print("-" * 45)
    for row in tx2["by_year"]:
        print(f"{row['year']:>6} | ${row['st_gain']:>9,.0f} | ${row['lt_gain']:>9,.0f} | ${row['tax']:>9,.0f}")
    print(f"{'TOTAL':>6} |             |             | ${tx2['total_tax']:>9,.0f}")


if __name__ == "__main__":
    main()
