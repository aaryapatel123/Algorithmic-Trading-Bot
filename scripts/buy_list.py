#!/usr/bin/env python3
"""Fresh buy-list calculator — "I have $X, what do I buy?"

Asks for a portfolio value and prints the dollar amount and whole-share count
to BUY for each equity plus the gold sleeve. Unlike signal_generator.py this
assumes NO existing holdings — it computes the target allocation from scratch
(every line is a buy) and keeps no state. Use it for a fresh deployment or a
quick sanity check of the current allocation.

Reuses the definitive strategy logic from signal_generator.py:
    keep8 + smooth 0.5 + cap25 + 10% GLD sleeve  →  85% equity / 10% gold / 5% cash

Usage:
    python scripts/buy_list.py
    python scripts/buy_list.py --portfolio-value 50000
    python scripts/buy_list.py 50000
"""
from __future__ import annotations

import argparse
import datetime
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.signal_generator import (
    CASH_BUFFER,
    SLEEVE_ASSETS,
    SLEEVE_PCT,
    TOP_N,
    compute_signal,
    fetch_prices,
    sleeve_weights,
)
from src.strategy.combined_momentum import _FALLBACK_UNIVERSE


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

def build_buy_list(portfolio_value: float, whole_shares: bool = False) -> dict:
    """Compute the from-scratch target allocation and per-line buy quantities.

    Default (fractional/dollar orders): each line is bought for its exact dollar
    target, so cash lands at exactly the 5% buffer. Set ``whole_shares=True`` for
    integer-share orders — then high-priced names round down and the leftover
    rolls into cash, so the achieved cash weight can exceed 5% (worse on small
    accounts). In both modes the reported ``weight`` is the *achieved* weight
    (buy_dollars / portfolio_value), so it can never disagree with ``buy_dollars``.

    Returns {"rows": [...], "cash_dollars", "invested_dollars", "whole_shares"}.
    Each row: symbol, kind ("equity"|"gold"), weight, target_dollars, price,
    shares (float; integer when whole_shares), buy_dollars.
    """
    fetch_symbols = list(_FALLBACK_UNIVERSE) + [s for s in SLEEVE_ASSETS if s not in _FALLBACK_UNIVERSE]
    prices = fetch_prices(fetch_symbols)

    equity_prices = {s: p for s, p in prices.items() if s not in SLEEVE_ASSETS}
    if len(equity_prices) < TOP_N:
        raise RuntimeError("not enough equity price data to build a buy list")

    missing_sleeve = [s for s in SLEEVE_ASSETS if s not in prices]
    if missing_sleeve:
        raise RuntimeError(f"no price data for sleeve asset(s) {missing_sleeve}")

    # Fresh allocation: no prior holdings, so smoothing is a no-op and every
    # target is a buy. compute_signal already applies hysteresis/inv-vol/cap.
    signal = compute_signal(equity_prices, {})
    if not signal:
        raise RuntimeError("could not compute equity signal")

    target_weights = {**signal["target_weights"], **sleeve_weights()}
    sleeve_set = set(SLEEVE_ASSETS)

    rows = []
    invested = 0.0
    for sym, weight in target_weights.items():
        target_dollars = weight * portfolio_value
        price = float(prices[sym].iloc[-1])
        if price <= 0:
            shares = 0.0
        elif whole_shares:
            shares = float(math.floor(target_dollars / price))
        else:
            shares = target_dollars / price
        buy_dollars = shares * price
        invested += buy_dollars
        rows.append({
            "symbol": sym,
            "kind": "gold" if sym in sleeve_set else "equity",
            "weight": buy_dollars / portfolio_value,  # achieved, not intended
            "target_dollars": target_dollars,
            "price": price,
            "shares": shares,
            "buy_dollars": buy_dollars,
        })

    # Sort: equities by weight desc, then gold last.
    rows.sort(key=lambda r: (r["kind"] == "gold", -r["weight"]))

    return {
        "rows": rows,
        "invested_dollars": invested,
        "cash_dollars": portfolio_value - invested,
        "whole_shares": whole_shares,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_buy_list(result: dict, portfolio_value: float, as_of: datetime.date) -> str:
    rows = result["rows"]
    whole = result["whole_shares"]
    mode = "whole shares" if whole else "fractional / dollar orders"
    lines = []
    lines.append("=" * 64)
    lines.append(f"  BUY LIST — {as_of.strftime('%B %d, %Y')}")
    lines.append(f"  {(1 - CASH_BUFFER - SLEEVE_PCT) * 100:.0f}% equity / "
                 f"{SLEEVE_PCT * 100:.0f}% gold / {CASH_BUFFER * 100:.0f}% cash"
                 f"   ({mode})")
    lines.append("=" * 64)
    lines.append(f"  Portfolio value : ${portfolio_value:>12,.2f}")
    lines.append("")
    lines.append(f"  {'Symbol':<8}{'Alloc%':>8}{'Buy $':>12}{'Shares':>11}{'Price':>11}")
    lines.append("  " + "-" * 50)

    for r in rows:
        tag = "  ◀ gold" if r["kind"] == "gold" else ""
        pct = f"{r['weight'] * 100:.1f}%"
        buy = f"${r['buy_dollars']:,.2f}"
        shares = f"{r['shares']:,.0f}" if whole else f"{r['shares']:,.4f}"
        price = f"${r['price']:,.2f}"
        lines.append(
            f"  {r['symbol']:<8}{pct:>8}{buy:>12}{shares:>11}{price:>11}{tag}"
        )

    lines.append("  " + "-" * 50)
    cash_pct = f"{result['cash_dollars'] / portfolio_value * 100:.1f}%"
    cash_amt = f"${result['cash_dollars']:,.2f}"
    lines.append(f"  {'CASH':<8}{cash_pct:>8}{cash_amt:>12}")
    lines.append("")
    lines.append(f"  Invested : ${result['invested_dollars']:>12,.2f}")
    cash_note = ("5% buffer + whole-share rounding" if whole
                 else "exactly the 5% buffer")
    lines.append(f"  Cash     : ${result['cash_dollars']:>12,.2f}  ({cash_note})")
    lines.append("=" * 64)
    if whole:
        lines.append("  Whole-share mode rounds down, so cash can exceed 5% (worst on")
        lines.append("  small accounts / high-priced names). Drop --whole-shares to use")
        lines.append("  Robinhood fractional/dollar orders and hit the targets exactly.")
    else:
        lines.append("  Place each 'Buy $' as a Robinhood dollar order (fractional shares).")
        lines.append("  Use --whole-shares if you want integer share counts instead.")
    lines.append("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_value(args: argparse.Namespace) -> float:
    raw = args.portfolio_value if args.portfolio_value is not None else args.value
    if raw is None:
        try:
            raw = input("Enter current portfolio value ($): ").replace(",", "").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nError: portfolio value required.")
            sys.exit(1)
    try:
        value = float(str(raw).replace(",", ""))
    except ValueError:
        print(f"Error: '{raw}' is not a valid number.")
        sys.exit(1)
    if value <= 0:
        print("Error: portfolio value must be positive.")
        sys.exit(1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh buy-list calculator (equities + gold)")
    parser.add_argument("value", nargs="?", type=str, help="Portfolio value in USD (positional)")
    parser.add_argument("--portfolio-value", type=str, help="Portfolio value in USD")
    parser.add_argument("--whole-shares", action="store_true",
                        help="Round to integer shares (default: fractional/dollar orders)")
    args = parser.parse_args()

    portfolio_value = _resolve_value(args)

    print(f"\nFetching prices for {len(_FALLBACK_UNIVERSE) + len(SLEEVE_ASSETS)} symbols…")
    try:
        result = build_buy_list(portfolio_value, whole_shares=args.whole_shares)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(format_buy_list(result, portfolio_value, datetime.date.today()))


if __name__ == "__main__":
    main()
