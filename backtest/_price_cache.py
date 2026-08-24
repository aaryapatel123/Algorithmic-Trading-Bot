"""Local disk cache for yfinance price downloads.

Backtests in this project pull ~160 symbols (SPY, AGG, and the 158-stock
universe) from yfinance on every run. Re-downloading on each strategy sweep is
the dominant cost. This module memoises the *cleaned* OHLCV frame to a local
pickle keyed by (symbol, start, end, auto_adjust), so the first run populates the
cache and every subsequent run reads from disk.

Cache files live in ``backtest/.price_cache/`` (git-ignored). Delete that folder
to force a fresh pull.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import pathlib
import pickle

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_DIR = pathlib.Path(__file__).resolve().parent / ".price_cache"


def _key(symbol: str, start: datetime.date, end: datetime.date, auto_adjust: bool) -> pathlib.Path:
    raw = f"{symbol}|{start}|{end}|{auto_adjust}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{symbol}_{digest}.pkl"


def get_prices(
    symbol: str,
    start: datetime.date,
    end: datetime.date,
    *,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Return a cleaned OHLCV frame (lowercase columns, flat index) for ``symbol``.

    Reads from the local pickle cache when present, otherwise downloads via
    yfinance and writes the cache. Raises ``ValueError`` if no data is available.
    """
    path = _key(symbol, start, end, auto_adjust)
    if path.exists():
        try:
            df = pickle.loads(path.read_bytes())
            if not df.empty:
                return df.copy()
        except Exception as exc:  # corrupt cache entry — fall through to re-download
            logger.warning("Cache read failed for %s (%s); re-downloading", symbol, exc)

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=auto_adjust)
    if df.empty:
        raise ValueError(f"No data for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(
        columns={"Open": "open", "High": "high", "Low": "low",
                 "Close": "close", "Volume": "volume"}
    )
    df.index = pd.to_datetime(df.index)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(pickle.dumps(df))
    except Exception as exc:
        logger.warning("Cache write failed for %s (%s)", symbol, exc)

    return df.copy()
