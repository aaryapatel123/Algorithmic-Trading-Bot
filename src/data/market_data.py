from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.data.base import DataProvider

logger = logging.getLogger(__name__)

_ALPACA_TIMEFRAME_MAP = {
    "1D": TimeFrame.Day,
    "1H": TimeFrame.Hour,
    "1Min": TimeFrame.Minute,
}
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds


class AlpacaDataProvider(DataProvider):
    def __init__(self, client: StockHistoricalDataClient) -> None:
        self._client = client

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 300,
    ) -> pd.DataFrame:
        timeframe_key = _ALPACA_TIMEFRAME_MAP.get(timeframe, TimeFrame.Day)
        bars = self._fetch_with_retry(symbol, timeframe_key, limit)

        if bars is not None and not bars.empty:
            return bars

        logger.warning(
            "Alpaca returned no data for %s, falling back to yfinance", symbol
        )
        return self._yfinance_fallback(symbol, limit)

    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: TimeFrame,
        limit: int,
    ) -> Optional[pd.DataFrame]:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
                barset = self._client.get_stock_bars(request)
                bars = barset[symbol]
                if not bars:
                    return None

                df = pd.DataFrame(
                    {
                        "open": [b.open for b in bars],
                        "high": [b.high for b in bars],
                        "low": [b.low for b in bars],
                        "close": [b.close for b in bars],
                        "volume": [b.volume for b in bars],
                    },
                    index=pd.to_datetime([b.timestamp for b in bars], utc=True),
                )
                df.sort_index(inplace=True)
                logger.debug(
                    "Fetched %d bars for %s from Alpaca", len(df), symbol
                )
                return df
            except Exception as exc:
                logger.warning(
                    "Alpaca data fetch attempt %d/%d failed for %s: %s",
                    attempt,
                    _MAX_RETRIES,
                    symbol,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY * attempt)
        return None

    def _yfinance_fallback(self, symbol: str, limit: int) -> pd.DataFrame:
        period = "1y" if limit <= 252 else "2y"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            logger.error("yfinance also returned no data for %s", symbol)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )[["open", "high", "low", "close", "volume"]]

        if not df.index.tzinfo:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.tail(limit)
        logger.debug("Fetched %d bars for %s via yfinance", len(df), symbol)
        return df

    def get_latest_price(self, symbol: str) -> float:
        df = self.get_historical_bars(symbol, timeframe="1D", limit=1)
        if df.empty:
            raise ValueError(f"Could not retrieve latest price for {symbol}")
        return float(df["close"].iloc[-1])
