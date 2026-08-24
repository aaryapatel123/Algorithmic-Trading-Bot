"""Monthly rebalance signal generator — Robinhood Roth IRA.

Computes the definitive strategy signal (keep8 + smooth 0.5 + cap25 + 10% GLD
sleeve) and prints a clear human-readable trade list. Optionally emails it via
Gmail.

The equity book gets 85% of the portfolio (1 - 5% cash - 10% gold). The gold
sleeve is static and equal-weighted — never smoothed, always exactly 10%.

Usage:
    python scripts/signal_generator.py
    python scripts/signal_generator.py --portfolio-value 50000
    python scripts/signal_generator.py --portfolio-value 50000 --email you@gmail.com

State is persisted in scripts/.signal_state.json between runs so that the
EWMA weight smoothing carries forward correctly across months.

Email setup (optional):
    Create a .env file in the project root (or export env vars directly):
        GMAIL_USER=you@gmail.com
        GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx   # Google account app password
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import smtplib
import sys
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest._hysteresis import apply_hysteresis, smooth_weights
from backtest._sector_map import SECTOR_MAP
from src.strategy.combined_momentum import _FALLBACK_UNIVERSE

# ---------------------------------------------------------------------------
# Strategy config — mirrors the definitive backtest pick: keep8 + smooth 0.5
# + sector_cap 80% + cat_stop 25% (Session 13)
# ---------------------------------------------------------------------------
TOP_N            = 5
KEEP_N           = 8       # hysteresis buffer: retain a held name while ranked <= 8
MAX_WEIGHT       = 0.25    # 25% position cap (water-filling)
ALPHA            = 0.5     # EWMA smoothing: 1.0 = no smoothing, lower = stickier
CASH_BUFFER      = 0.05    # 5% kept in cash every rebalance
SLEEVE_PCT       = 0.10    # 10% permanent gold sleeve (Session 12 baseline)
SLEEVE_ASSETS    = ("GLD",)  # static, equal-weighted — never smoothed
MOM_DAYS         = 252     # ~12 months of trading days for momentum ranking
VOL_DAYS         = 20      # rolling window for inverse-vol weighting
LOOKBACK_CAL     = 420     # calendar days of history to fetch (MOM_DAYS + buffer)
SECTOR_CAP         = 0.80    # max portfolio weight any single GICS sector may hold
CAT_STOP_WARNING   = 0.20    # warn when a held name is >20% below its entry price
UNIVERSE_CACHE_TTL = 180     # days before re-scraping S&P 500 membership from Wikipedia


def _state_file() -> pathlib.Path:
    """Where to persist EWMA/hysteresis state across runs.

    In a frozen (PyInstaller) build, ``__file__`` resolves into an ephemeral
    ``_MEI`` temp dir that is deleted on exit — persisting there would silently
    drop the cross-month smoothing state. Store next to the executable instead.
    """
    base = (pathlib.Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else pathlib.Path(__file__).resolve().parent)
    return base / ".signal_state.json"


STATE_FILE = _state_file()
UNIVERSE_CACHE_FILE = STATE_FILE.parent / ".universe_cache.json"


# ---------------------------------------------------------------------------
# State — persists prev weights and last rebalance month
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "prev_weights": {},
        "last_rebalance": None,
        "portfolio_value": None,
        "entry_prices": {},
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Universe — S&P 500 large-caps, refreshed from Wikipedia every 6 months
# ---------------------------------------------------------------------------

def load_universe() -> list[str]:
    """Return the equity universe, refreshing from Wikipedia if the cache is stale.

    Cache TTL is UNIVERSE_CACHE_TTL days (180 = ~6 months). On any fetch failure
    the existing cache is kept; if no cache exists at all, falls back to
    _FALLBACK_UNIVERSE so the script always has something to run against.
    """
    today = datetime.date.today()

    if UNIVERSE_CACHE_FILE.exists():
        try:
            cached = json.loads(UNIVERSE_CACHE_FILE.read_text())
            cached_date = datetime.date.fromisoformat(cached["as_of"])
            age_days = (today - cached_date).days
            if age_days < UNIVERSE_CACHE_TTL:
                return cached["symbols"]
            print(f"  Universe cache is {age_days} days old — refreshing from Wikipedia…")
        except Exception:
            pass

    try:
        import io
        import urllib.request
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read()
        tables = pd.read_html(io.BytesIO(html), match="Symbol")
        df = tables[0]
        symbols = (
            df["Symbol"]
            .str.replace(".", "-", regex=False)
            .str.strip()
            .str.upper()
            .tolist()
        )
        symbols = [s for s in symbols if isinstance(s, str) and s]
        if len(symbols) < 50:
            raise ValueError(f"Wikipedia returned only {len(symbols)} tickers")

        # Keep only symbols we have sector data for — screens out corporate-action
        # artifacts and untested new entrants until SECTOR_MAP is updated.
        symbols = [s for s in symbols if s in SECTOR_MAP]
        if len(symbols) < 50:
            raise ValueError(f"Only {len(symbols)} symbols survived sector-map filter")

        UNIVERSE_CACHE_FILE.write_text(json.dumps(
            {"as_of": today.isoformat(), "symbols": symbols}, indent=2
        ))
        print(f"  Universe refreshed: {len(symbols)} large-cap symbols in SECTOR_MAP (saved to cache).")
        return symbols
    except Exception as exc:
        if UNIVERSE_CACHE_FILE.exists():
            try:
                cached = json.loads(UNIVERSE_CACHE_FILE.read_text())
                print(f"  Wikipedia fetch failed ({exc}) — using stale cache ({cached['as_of']}).")
                return cached["symbols"]
            except Exception:
                pass
        print(f"  Wikipedia fetch failed ({exc}) — using built-in fallback universe.")
        return list(_FALLBACK_UNIVERSE)


# ---------------------------------------------------------------------------
# Price data — always fetches fresh (no disk cache for live signals)
# ---------------------------------------------------------------------------

def fetch_prices(symbols: list[str], lookback_days: int = LOOKBACK_CAL) -> dict[str, pd.Series]:
    """Return {symbol: close_price_series} for all symbols with enough history."""
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=lookback_days)

    prices: dict[str, pd.Series] = {}
    failed: list[str] = []

    data = yf.download(
        symbols,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if data.empty:
        return prices

    # yfinance returns MultiIndex columns when >1 symbol
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]].rename(columns={"Close": symbols[0]})

    for sym in symbols:
        if sym not in close.columns:
            failed.append(sym)
            continue
        s = close[sym].dropna()
        if len(s) < MOM_DAYS + VOL_DAYS:
            failed.append(sym)
            continue
        prices[sym] = s

    if failed:
        pass  # silently skip — same behaviour as backtest

    return prices


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_momentum(prices: dict[str, pd.Series]) -> dict[str, float]:
    """12-month (252 trading-day) total return for each symbol."""
    mom: dict[str, float] = {}
    for sym, s in prices.items():
        if len(s) < MOM_DAYS + 1:
            continue
        ret = float(s.iloc[-1] / s.iloc[-(MOM_DAYS + 1)] - 1.0)
        mom[sym] = ret
    return mom


def compute_inv_vol(symbols: list[str], prices: dict[str, pd.Series]) -> dict[str, float]:
    """20-day rolling volatility (annualised std of daily returns) for each symbol."""
    vols: dict[str, float] = {}
    for sym in symbols:
        s = prices[sym]
        daily = s.pct_change().dropna()
        if len(daily) < VOL_DAYS:
            continue
        v = float(daily.iloc[-VOL_DAYS:].std()) * (252 ** 0.5)
        if v > 0:
            vols[sym] = v
    return vols


def cap_weights(raw: dict[str, float], investable: float, max_w: float) -> dict[str, float]:
    """Water-fill cap: no single name > max_w. Excess redistributed to uncapped names."""
    if max_w >= 1.0 or not raw:
        return dict(raw)
    weights = dict(raw)
    capped: set[str] = set()
    for _ in range(len(weights)):
        over = [s for s, w in weights.items() if w > max_w + 1e-9 and s not in capped]
        if not over:
            break
        for s in over:
            capped.add(s)
            weights[s] = max_w
        used = sum(weights[s] for s in capped)
        remaining = investable - used
        free = [s for s in weights if s not in capped]
        base = sum(raw[s] for s in free)
        if remaining <= 0 or base <= 0 or not free:
            break
        for s in free:
            weights[s] = raw[s] / base * remaining
    return weights


def apply_sector_cap(
    weights: dict[str, float],
    sector_map: dict[str, str],
    cap: float,
) -> dict[str, float]:
    """Scale down any sector whose total portfolio weight exceeds cap.

    Freed weight becomes cash — no redistribution to other names.
    Pass only equity weights (sleeve assets excluded).
    """
    if cap >= 1.0 or not weights:
        return dict(weights)
    sector_totals: dict[str, float] = {}
    for sym, w in weights.items():
        s = sector_map.get(sym, "Unknown")
        sector_totals[s] = sector_totals.get(s, 0.0) + w
    result = dict(weights)
    for sector, total in sector_totals.items():
        if total > cap + 1e-9:
            scale = cap / total
            for sym in result:
                if sector_map.get(sym, "Unknown") == sector:
                    result[sym] = result[sym] * scale
    return result


def check_cat_stop_warnings(
    target_weights: dict[str, float],
    entry_prices: dict[str, float],
    current_prices: dict[str, pd.Series],
    threshold: float = CAT_STOP_WARNING,
) -> list[dict]:
    """Return held names that have fallen more than threshold below their entry price."""
    warnings: list[dict] = []
    for sym, entry_px in entry_prices.items():
        if target_weights.get(sym, 0.0) <= 0:
            continue
        if sym not in current_prices or entry_px <= 0:
            continue
        current_px = float(current_prices[sym].iloc[-1])
        drawdown = current_px / entry_px - 1.0
        if drawdown < -threshold:
            warnings.append({
                "symbol": sym,
                "entry_price": entry_px,
                "current_price": round(current_px, 2),
                "drawdown_pct": round(drawdown * 100, 1),
            })
    return warnings


def compute_signal(
    prices: dict[str, pd.Series],
    prev_weights: dict[str, float],
) -> dict:
    """Full signal: momentum rank → hysteresis selection → inv-vol weights → cap → smooth."""
    mom = compute_momentum(prices)
    if not mom:
        return {}

    # Rank all symbols by 12-month momentum, descending (price data required)
    ranked = [s for s in sorted(mom.keys(), key=lambda s: mom[s], reverse=True) if s in prices]

    # Hysteresis: retain held names while ranked <= KEEP_N, fill remainder from top
    held = {s for s, w in prev_weights.items() if w > 0}
    selected: list[str] = apply_hysteresis(ranked, held, TOP_N, KEEP_N)

    if not selected:
        return {}

    vols = compute_inv_vol(selected, prices)
    valid = [s for s in selected if s in vols]
    if not valid:
        return {}

    # Inverse-vol fractions (sum to 1.0)
    total_inv = sum(1.0 / vols[s] for s in valid)
    fractions = {s: (1.0 / vols[s]) / total_inv for s in valid}

    # Scale to equity budget (portfolio minus cash buffer and gold sleeve)
    investable = 1.0 - CASH_BUFFER - SLEEVE_PCT
    raw_weights = {s: fractions[s] * investable for s in valid}

    # Apply 25% per-name cap (water-filling)
    capped = cap_weights(raw_weights, investable, MAX_WEIGHT)

    # Apply sector cap: SECTOR_CAP is defined as % of the equity book.
    # Convert to a portfolio-fraction cap (e.g. 80% × 85% equity = 68% of portfolio).
    effective_sector_cap = SECTOR_CAP * investable
    sector_capped = apply_sector_cap(capped, SECTOR_MAP, effective_sector_cap)

    # Fill freed equity capacity with the best momentum stock from a different sector.
    # The sector cap clips ~20% of the equity book to cash; invest it instead.
    freed = investable - sum(sector_capped.values())
    diversification_pick: str | None = None
    if freed > 0.005:
        at_cap_sectors = {
            SECTOR_MAP.get(s, "Unknown")
            for s, w in sector_capped.items()
            if w >= effective_sector_cap - 1e-9
        }
        # Broaden: any sector whose total weight is near the cap
        sector_totals_check: dict[str, float] = {}
        for s, w in sector_capped.items():
            sec = SECTOR_MAP.get(s, "Unknown")
            sector_totals_check[sec] = sector_totals_check.get(sec, 0.0) + w
        at_cap_sectors |= {
            sec for sec, w in sector_totals_check.items()
            if w >= effective_sector_cap - 1e-9
        }
        already_selected = set(sector_capped)
        for candidate in ranked:
            if candidate in already_selected:
                continue
            if candidate not in prices:
                continue
            if SECTOR_MAP.get(candidate, "Unknown") in at_cap_sectors:
                continue
            alloc = min(freed, MAX_WEIGHT)
            sector_capped = {**sector_capped, candidate: alloc}
            diversification_pick = candidate
            break

    # Apply EWMA smoothing toward previous weights
    smoothed = smooth_weights(sector_capped, prev_weights, ALPHA)

    # Re-apply sector cap after smoothing — EWMA can pull weights back above the cap
    # when prev_weights were over-cap (e.g. first run after parameter change).
    smoothed = apply_sector_cap(smoothed, SECTOR_MAP, effective_sector_cap)

    # Compute sector breakdown as fraction of equity book for reporting
    sector_totals_portfolio: dict[str, float] = {}
    for sym, w in smoothed.items():
        s = SECTOR_MAP.get(sym, "Unknown")
        sector_totals_portfolio[s] = sector_totals_portfolio.get(s, 0.0) + w

    # Express as % of equity book so the 80% cap threshold reads naturally
    sector_totals_equity = {
        s: w / investable for s, w in sector_totals_portfolio.items()
    }

    all_selected = list(valid) + ([diversification_pick] if diversification_pick else [])
    all_momentum = {s: mom[s] for s in all_selected if s in mom}

    return {
        "selected": all_selected,
        "momentum": all_momentum,
        "target_weights": smoothed,
        "ranked_top10": ranked[:10],
        "ranked_momentum": {s: mom[s] for s in ranked[:10]},
        "sector_totals": sector_totals_equity,
        "equity_budget": investable,
        "diversification_pick": diversification_pick,
    }


# ---------------------------------------------------------------------------
# Defensive sleeve — static, equal-weighted, never smoothed
# ---------------------------------------------------------------------------

def sleeve_weights() -> dict[str, float]:
    """Permanent gold sleeve: equal-weighted across SLEEVE_ASSETS, total SLEEVE_PCT.

    Returns a fresh dict each call. The sleeve is intentionally not smoothed and
    not part of the momentum ranking — it always sits at exactly SLEEVE_PCT.
    """
    if SLEEVE_PCT <= 0.0 or not SLEEVE_ASSETS:
        return {}
    per = SLEEVE_PCT / len(SLEEVE_ASSETS)
    return {asset: per for asset in SLEEVE_ASSETS}


# ---------------------------------------------------------------------------
# Trade diff — what to buy and sell
# ---------------------------------------------------------------------------

def compute_trades(
    target_weights: dict[str, float],
    prev_weights: dict[str, float],
    portfolio_value: float,
    current_prices: dict[str, pd.Series],
) -> dict:
    all_symbols = set(target_weights) | set(prev_weights)
    sells, buys, holds = [], [], []

    for sym in all_symbols:
        target_frac = target_weights.get(sym, 0.0)
        current_frac = prev_weights.get(sym, 0.0)
        delta_frac = target_frac - current_frac
        delta_dollars = delta_frac * portfolio_value
        target_dollars = target_frac * portfolio_value

        price = float(current_prices[sym].iloc[-1]) if sym in current_prices else None
        shares = round(delta_dollars / price) if price and price > 0 else None

        entry = {
            "symbol": sym,
            "current_weight": round(current_frac * 100, 1),
            "target_weight": round(target_frac * 100, 1),
            "delta_dollars": round(delta_dollars),
            "target_dollars": round(target_dollars),
            "price": round(price, 2) if price else None,
            "shares": shares,
        }
        if delta_frac < -0.005:
            sells.append(entry)
        elif delta_frac > 0.005:
            buys.append(entry)
        else:
            holds.append(entry)

    sells.sort(key=lambda x: x["delta_dollars"])
    buys.sort(key=lambda x: x["delta_dollars"], reverse=True)

    return {"sells": sells, "buys": buys, "holds": holds}


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    signal: dict,
    trades: dict,
    portfolio_value: float,
    as_of: datetime.date,
    is_rebalance_month: bool,
    cat_stop_warnings: list[dict] | None = None,
) -> str:
    lines = []
    lines.append("=" * 62)
    lines.append(f"  MOMENTUM SIGNAL — {as_of.strftime('%B %d, %Y')}")
    if not is_rebalance_month:
        lines.append("  STATUS: Preview only (already rebalanced this month)")
    else:
        lines.append("  STATUS: REBALANCE DUE — execute trades in Robinhood")
    lines.append("=" * 62)
    lines.append(f"  Portfolio value : ${portfolio_value:>12,.2f}")
    lines.append(f"  Cash target     : ${portfolio_value * CASH_BUFFER:>12,.2f}  ({CASH_BUFFER*100:.0f}%)")
    if SLEEVE_PCT > 0 and SLEEVE_ASSETS:
        sleeve_str = "/".join(SLEEVE_ASSETS)
        lines.append(f"  Gold sleeve     : ${portfolio_value * SLEEVE_PCT:>12,.2f}  ({SLEEVE_PCT*100:.0f}%  {sleeve_str})")
    lines.append(f"  Sector cap      : {SECTOR_CAP*100:.0f}% of equity  |  Cat-stop warn : >{CAT_STOP_WARNING*100:.0f}% drawdown from entry")
    lines.append("")

    # Sector concentration (weights expressed as % of equity book)
    sector_totals = signal.get("sector_totals", {})
    if sector_totals:
        lines.append("── Sector Concentration (% of equity book) ─────────────────")
        for sector, equity_frac in sorted(sector_totals.items(), key=lambda x: -x[1]):
            flag = "  ⚠ AT CAP" if equity_frac >= SECTOR_CAP - 0.001 else ""
            lines.append(f"  {sector:<28}  {equity_frac*100:>5.1f}%{flag}")
        lines.append("")

    # Cat-stop warnings
    if cat_stop_warnings:
        lines.append("── ⚠  CAT-STOP WARNINGS (manual exit recommended) ─────────")
        for w in cat_stop_warnings:
            lines.append(
                f"  {w['symbol']:<6}  entry ${w['entry_price']:>8.2f}  "
                f"now ${w['current_price']:>8.2f}  "
                f"({w['drawdown_pct']:>+.1f}%)  EXIT BEFORE NEXT REBALANCE"
            )
        lines.append("")

    # Top-10 momentum ranking
    lines.append("── Top-10 Momentum Ranking (12-month return) ──────────────")
    for i, sym in enumerate(signal["ranked_top10"], 1):
        arrow = "  ◀ SELECTED" if sym in signal["selected"] else ""
        lines.append(f"  {i:>2}. {sym:<6}  {signal['ranked_momentum'][sym]*100:>+7.1f}%{arrow}")
    lines.append("")

    # Sells
    if trades["sells"]:
        lines.append("── SELL ────────────────────────────────────────────────────")
        for t in trades["sells"]:
            shares_str = f"{abs(t['shares'])} shares" if t["shares"] is not None else "N/A shares"
            lines.append(
                f"  SELL  {t['symbol']:<6}  {t['current_weight']:>5.1f}% → {t['target_weight']:>5.1f}%"
                f"   ${abs(t['delta_dollars']):>9,.0f}   {shares_str} @ ${t['price']:,.2f}"
            )
    else:
        lines.append("── No sells required ───────────────────────────────────────")

    lines.append("")

    # Buys
    if trades["buys"]:
        lines.append("── BUY ─────────────────────────────────────────────────────")
        for t in trades["buys"]:
            shares_str = f"{abs(t['shares'])} shares" if t["shares"] is not None else "N/A shares"
            lines.append(
                f"  BUY   {t['symbol']:<6}  {t['current_weight']:>5.1f}% → {t['target_weight']:>5.1f}%"
                f"   ${abs(t['delta_dollars']):>9,.0f}   {shares_str} @ ${t['price']:,.2f}"
            )
    else:
        lines.append("── No new buys required ────────────────────────────────────")

    lines.append("")

    # Holds
    if trades["holds"]:
        lines.append("── HOLD (no change) ────────────────────────────────────────")
        for t in trades["holds"]:
            lines.append(
                f"  HOLD  {t['symbol']:<6}  {t['target_weight']:>5.1f}%"
                f"   ${t['target_dollars']:>9,.0f}"
            )

    lines.append("")
    lines.append("── Final target portfolio ──────────────────────────────────")
    all_pos = {**{t["symbol"]: t for t in trades["sells"]},
               **{t["symbol"]: t for t in trades["buys"]},
               **{t["symbol"]: t for t in trades["holds"]}}
    div_pick = signal.get("diversification_pick")
    for sym in signal["selected"]:
        t = all_pos.get(sym, {})
        sector = SECTOR_MAP.get(sym, "")
        label = f"  (diversification — {sector})" if sym == div_pick else ""
        lines.append(
            f"  {sym:<6}  {t.get('target_weight', 0):>5.1f}%   ${t.get('target_dollars', 0):>9,.0f}{label}"
        )
    for sym in SLEEVE_ASSETS:
        t = all_pos.get(sym, {})
        lines.append(
            f"  {sym:<6}  {t.get('target_weight', 0):>5.1f}%   ${t.get('target_dollars', 0):>9,.0f}  (gold sleeve)"
        )
    cash_dollars = portfolio_value * CASH_BUFFER
    lines.append(f"  {'CASH':<6}  {CASH_BUFFER*100:>5.1f}%   ${cash_dollars:>9,.0f}")
    lines.append("=" * 62)
    lines.append("  Execute sells FIRST in Robinhood, then buys.")
    lines.append("  Use limit orders within the bid/ask spread.")
    lines.append("=" * 62)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def send_email(subject: str, body: str, to_addrs: list[str] | str) -> None:
    recipients = [to_addrs] if isinstance(to_addrs, str) else list(to_addrs)
    recipients = [r.strip() for r in recipients if r.strip()]
    if not recipients:
        print("  [email] No recipients — skipping email.")
        return

    gmail_user = os.environ.get("GMAIL_USER", "")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not gmail_user or not gmail_pass:
        print("  [email] GMAIL_USER / GMAIL_APP_PASSWORD not set — skipping email.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = ", ".join(recipients)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_user, gmail_pass)
            smtp.sendmail(gmail_user, recipients, msg.as_string())
        print(f"  [email] Signal sent to {', '.join(recipients)}")
    except Exception as exc:
        print(f"  [email] Failed to send: {exc}")


# ---------------------------------------------------------------------------
# .env loader (simple, no dependency on python-dotenv)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Monthly momentum rebalance signal")
    parser.add_argument("--portfolio-value", type=float, help="Current Roth IRA value in USD")
    parser.add_argument("--email", type=str, help="Email recipient(s), comma-separated")
    parser.add_argument("--force", action="store_true", help="Force signal even if already run this month")
    args = parser.parse_args()

    state = load_state()
    today = datetime.date.today()
    this_month = today.strftime("%Y-%m")

    # Resolve portfolio value: CLI > state file > prompt
    portfolio_value = args.portfolio_value or state.get("portfolio_value")
    if portfolio_value is None:
        try:
            portfolio_value = float(input("Enter current Roth IRA portfolio value ($): ").replace(",", ""))
        except (ValueError, EOFError):
            print("Error: portfolio value required.")
            sys.exit(1)
    portfolio_value = float(portfolio_value)
    state["portfolio_value"] = portfolio_value

    # Check if rebalance is due this month
    is_rebalance_month = (state.get("last_rebalance") != this_month) or args.force
    if not is_rebalance_month:
        print(f"Already rebalanced in {this_month}. Use --force to preview anyway.")
        # Continue to show a preview

    universe = load_universe()
    fetch_symbols = universe + [s for s in SLEEVE_ASSETS if s not in universe]
    print(f"\nFetching prices for {len(fetch_symbols)} symbols ({LOOKBACK_CAL} calendar days)…")
    prices = fetch_prices(fetch_symbols)
    print(f"  Got data for {len(prices)} symbols.\n")

    # Equity ranking excludes the sleeve assets — gold is static, not momentum-ranked.
    equity_prices = {s: p for s, p in prices.items() if s not in SLEEVE_ASSETS}

    if len(equity_prices) < TOP_N:
        print("Error: not enough price data.")
        sys.exit(1)

    missing_sleeve = [s for s in SLEEVE_ASSETS if s not in prices]
    if missing_sleeve:
        print(f"Error: no price data for sleeve asset(s) {missing_sleeve} — cannot size the gold sleeve.")
        sys.exit(1)

    prev_weights = {s: v for s, v in state.get("prev_weights", {}).items() if s in prices}

    signal = compute_signal(equity_prices, {s: v for s, v in prev_weights.items() if s not in SLEEVE_ASSETS})
    if not signal:
        print("Error: could not compute signal.")
        sys.exit(1)

    # Merge the static gold sleeve into the equity target (never smoothed).
    signal["target_weights"] = {**signal["target_weights"], **sleeve_weights()}

    trades = compute_trades(signal["target_weights"], prev_weights, portfolio_value, prices)

    # Cat-stop warnings: compare current prices to entry prices from last rebalance
    entry_prices = state.get("entry_prices", {})
    cat_warnings = check_cat_stop_warnings(prev_weights, entry_prices, prices)

    report = format_report(signal, trades, portfolio_value, today, is_rebalance_month, cat_warnings)
    print(report)

    if args.email:
        recipients = [e.strip() for e in args.email.split(",") if e.strip()]
        subject = f"[Trading Bot] Rebalance Signal — {today.strftime('%B %Y')}"
        send_email(subject, report, recipients)

    # Save state only if this is a real rebalance month
    if is_rebalance_month:
        state["prev_weights"] = signal["target_weights"]
        state["last_rebalance"] = this_month
        # Snapshot current prices for cat-stop tracking next month
        state["entry_prices"] = {
            sym: round(float(prices[sym].iloc[-1]), 4)
            for sym in signal["target_weights"]
            if sym in prices
        }
        save_state(state)
        print(f"\n  State saved to {STATE_FILE}")
    else:
        print("\n  [preview mode — state not updated]")


if __name__ == "__main__":
    main()
