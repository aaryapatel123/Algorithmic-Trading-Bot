"""Combined momentum strategy for live trading.

1. Dual Momentum  -- SPY vs AGG 12-month return decides stocks vs bonds
2. 200-day MA     -- SPY below MA200 forces bond regime
3. Top-N stocks   -- rank universe by 12-month momentum, equal-weight top N
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallback universe (~150 large-caps from S&P 500)
# ---------------------------------------------------------------------------
_FALLBACK_UNIVERSE: List[str] = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AFL",
    "AIG", "AMAT", "AMD", "AMGN", "AMZN", "ANET", "AVGO", "AXP", "BA", "BAC",
    "BDX", "BIIB", "BK", "BKNG", "BLK", "BMY", "BSX", "C", "CAT", "CB",
    "CCI", "CDNS", "CI", "CL", "CMCSA", "CME", "COF", "COP", "COST", "CRM",
    "CSCO", "CTAS", "CVS", "CVX", "D", "DD", "DE", "DHR", "DIS", "DUK",
    "ECL", "EL", "EMR", "EOG", "EW", "EXC", "F", "FDX", "FIS", "FISV",
    "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GPN", "GS", "HD", "HON",
    "IBM", "ICE", "INTC", "INTU", "ISRG", "ITW", "JNJ", "JPM", "KHC", "KLAC",
    "KO", "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD", "MCHP", "MCK",
    "MDLZ", "MDT", "MET", "META", "MMC", "MMM", "MO", "MRK", "MS", "MSFT",
    "MU", "NEE", "NFLX", "NKE", "NOC", "NOW", "NSC", "NVDA", "ORCL", "OXY",
    "PEP", "PFE", "PG", "PGR", "PLD", "PM", "PNC", "PSA", "PSX", "PYPL",
    "QCOM", "REGN", "ROP", "RTX", "SBUX", "SCHW", "SHW", "SLB", "SNPS", "SO",
    "SPG", "SPGI", "SYK", "T", "TDG", "TFC", "TGT", "TJX", "TMO", "TMUS",
    "TSLA", "TT", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VLO", "VRTX",
    "VZ", "WBA", "WFC", "WM", "WMT", "XEL", "XOM", "ZTS",
]


def get_universe(custom: Optional[List[str]] = None) -> List[str]:
    """Return the stock universe for momentum ranking.

    If *custom* is a non-empty list, use it directly (SPY/AGG excluded).
    Otherwise attempt to scrape S&P 500 tickers from Wikipedia,
    falling back to the hardcoded large-cap list on any failure.
    """
    excluded = {"SPY", "AGG"}

    if custom:
        return [s for s in custom if s not in excluded]

    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            match="Symbol",
        )
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
        if len(tickers) < 50:
            raise ValueError(f"Wikipedia returned only {len(tickers)} tickers")
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return [t for t in tickers if t not in excluded]
    except Exception as exc:
        logger.warning(
            "Failed to fetch S&P 500 from Wikipedia (%s) — using fallback (%d stocks)",
            exc,
            len(_FALLBACK_UNIVERSE),
        )
        return list(_FALLBACK_UNIVERSE)


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MomentumSignal:
    regime: str                   # "STOCKS", "BONDS(below_200MA)", "BONDS(bonds_preferred)"
    targets: Dict[str, float]     # symbol -> target weight (0.0 to 1.0)
    top_symbols: List[str]        # top-N stocks chosen (empty in bond regime)
    spy_mom: float                # SPY 12-month return
    agg_mom: float                # AGG 12-month return
    spy_price: float              # SPY current close
    spy_ma200: float              # SPY 200-day SMA
    timestamp: datetime


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class CombinedMomentumStrategy:
    """Compute a monthly MomentumSignal from market data.

    No Backtrader dependency — uses yfinance batch downloads directly.
    Designed to be called once per month at rebalance time.
    """

    def __init__(
        self,
        top_n: int = 3,
        momentum_days: int = 252,
        ma_period: int = 200,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if momentum_days <= 0:
            raise ValueError("momentum_days must be positive")
        if ma_period <= 0:
            raise ValueError("ma_period must be positive")

        self._top_n = top_n
        self._momentum_days = momentum_days
        self._ma_period = ma_period

    def compute_signal(self, universe: List[str]) -> MomentumSignal:
        """Download price history and return the target portfolio allocation.

        *universe* is the candidate stock list (SPY/AGG excluded — they are
        added internally for regime detection and as a stock candidate).
        """
        now = datetime.now(tz=timezone.utc)

        # Need enough history for both momentum and MA lookbacks, with
        # a 2× calendar-day multiplier to account for weekends/holidays.
        history_days = (max(self._momentum_days, self._ma_period) + 60) * 2
        period_str = f"{history_days}d"

        # ── Step 1: Download SPY + AGG for regime detection ──────────────
        spy_close, agg_close = self._download_regime_data(period_str)

        min_spy_bars = self._momentum_days + self._ma_period
        if spy_close is None or len(spy_close) < min_spy_bars:
            logger.error(
                "Insufficient SPY data (%d bars, need %d)",
                len(spy_close) if spy_close is not None else 0,
                min_spy_bars,
            )
            return self._bond_signal("BONDS(no_spy_data)", 0.0, 0.0, 0.0, 0.0, now)

        if agg_close is None or len(agg_close) < self._momentum_days:
            logger.error("Insufficient AGG data")
            return self._bond_signal("BONDS(no_agg_data)", 0.0, 0.0, 0.0, 0.0, now)

        spy_mom = float(spy_close.iloc[-1] / spy_close.iloc[-self._momentum_days] - 1.0)
        agg_mom = float(agg_close.iloc[-1] / agg_close.iloc[-self._momentum_days] - 1.0)
        spy_price = float(spy_close.iloc[-1])
        spy_ma200 = float(spy_close.rolling(self._ma_period).mean().iloc[-1])

        # ── Step 2: Dual momentum regime check ───────────────────────────
        in_stocks = spy_mom > agg_mom and spy_mom > 0.0

        # ── Step 3: 200-day MA crash filter ──────────────────────────────
        spy_above_ma = spy_price > spy_ma200
        if not spy_above_ma:
            in_stocks = False

        if not in_stocks:
            reason = (
                "below_200MA"
                if (spy_mom > agg_mom and spy_mom > 0.0 and not spy_above_ma)
                else "bonds_preferred"
            )
            logger.info(
                "Regime=BONDS(%s) | spy_mom=%.2f%% agg_mom=%.2f%% spy=%.2f ma200=%.2f",
                reason, spy_mom * 100, agg_mom * 100, spy_price, spy_ma200,
            )
            return MomentumSignal(
                regime=f"BONDS({reason})",
                targets={"AGG": 0.95},
                top_symbols=[],
                spy_mom=spy_mom,
                agg_mom=agg_mom,
                spy_price=spy_price,
                spy_ma200=spy_ma200,
                timestamp=now,
            )

        # ── Step 4: Stock regime — rank universe by 12m return ───────────
        # SPY is also a valid stock candidate (matches backtest logic)
        candidates = list(dict.fromkeys(["SPY"] + universe))

        logger.info("Downloading %d candidates for momentum ranking...", len(candidates))
        candidate_data = self._download_candidates(candidates, period_str)

        returns: Dict[str, float] = {}
        for sym in candidates:
            close = self._extract_close(candidate_data, sym)
            if close is None or len(close) < self._momentum_days:
                continue
            old_price = float(close.iloc[-self._momentum_days])
            if old_price <= 0:
                continue
            returns[sym] = float(close.iloc[-1] / old_price - 1.0)

        if not returns:
            logger.warning("No valid momentum returns — falling back to bonds")
            return self._bond_signal("BONDS(no_data)", spy_mom, agg_mom, spy_price, spy_ma200, now)

        ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in ranked[: self._top_n]]
        weight_per_stock = 0.95 / self._top_n
        targets = {sym: weight_per_stock for sym in top_symbols}

        logger.info(
            "Regime=STOCKS | top-%d=%s | spy_mom=%.2f%% agg_mom=%.2f%% spy=%.2f ma200=%.2f",
            self._top_n, top_symbols,
            spy_mom * 100, agg_mom * 100, spy_price, spy_ma200,
        )

        return MomentumSignal(
            regime="STOCKS",
            targets=targets,
            top_symbols=top_symbols,
            spy_mom=spy_mom,
            agg_mom=agg_mom,
            spy_price=spy_price,
            spy_ma200=spy_ma200,
            timestamp=now,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _download_regime_data(
        self, period: str
    ) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Download SPY and AGG closing prices."""
        try:
            data = yf.download(
                ["SPY", "AGG"],
                period=period,
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
            spy = self._extract_close(data, "SPY")
            agg = self._extract_close(data, "AGG")
            return spy, agg
        except Exception as exc:
            logger.error("Failed to download SPY/AGG: %s", exc)
            return None, None

    def _download_candidates(self, symbols: List[str], period: str) -> pd.DataFrame:
        """Batch-download closing prices for all candidate symbols."""
        try:
            return yf.download(
                symbols,
                period=period,
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            logger.error("Failed to download candidates: %s", exc)
            return pd.DataFrame()

    @staticmethod
    def _extract_close(data: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
        """Extract the Close series for *symbol* from a yfinance MultiIndex result."""
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data[symbol]["Close"].dropna()
            else:
                # Single-ticker download fallback
                close = data["Close"].dropna() if "Close" in data.columns else pd.Series(dtype=float)
            return close if len(close) > 0 else None
        except (KeyError, TypeError):
            return None

    def _bond_signal(
        self,
        regime: str,
        spy_mom: float,
        agg_mom: float,
        spy_price: float,
        spy_ma200: float,
        timestamp: datetime,
    ) -> MomentumSignal:
        return MomentumSignal(
            regime=regime,
            targets={"AGG": 0.95},
            top_symbols=[],
            spy_mom=spy_mom,
            agg_mom=agg_mom,
            spy_price=spy_price,
            spy_ma200=spy_ma200,
            timestamp=timestamp,
        )
