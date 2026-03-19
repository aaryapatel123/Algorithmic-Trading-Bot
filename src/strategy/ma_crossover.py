from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from src.strategy.base import Signal, Strategy

logger = logging.getLogger(__name__)


class MACrossoverStrategy(Strategy):
    def __init__(self, short_period: int, long_period: int) -> None:
        if short_period <= 0:
            raise ValueError("short_period must be positive")
        if long_period <= 0:
            raise ValueError("long_period must be positive")
        if short_period >= long_period:
            raise ValueError("short_period must be less than long_period")

        self._short_period = short_period
        self._long_period = long_period

    def compute_signal(self, symbol: str, bars: pd.DataFrame) -> Signal:
        if bars.empty or "close" not in bars.columns:
            logger.warning("Empty or invalid bars for %s — returning HOLD", symbol)
            return self._hold(symbol, 0.0, 0.0)

        close = bars["close"]

        if len(close) < self._long_period + 1:
            logger.warning(
                "Insufficient data for %s (%d bars, need %d) — returning HOLD",
                symbol,
                len(close),
                self._long_period + 1,
            )
            return self._hold(symbol, 0.0, 0.0)

        short_ma = close.rolling(self._short_period).mean()
        long_ma = close.rolling(self._long_period).mean()

        # Drop leading NaN from both series
        valid = short_ma.notna() & long_ma.notna()
        if valid.sum() < 2:
            logger.warning("Not enough valid MA values for %s — returning HOLD", symbol)
            return self._hold(symbol, float(short_ma.iloc[-1]), float(long_ma.iloc[-1]))

        short_now = float(short_ma.iloc[-1])
        short_prev = float(short_ma[valid].iloc[-2])
        long_now = float(long_ma.iloc[-1])
        long_prev = float(long_ma[valid].iloc[-2])

        timestamp = datetime.now(tz=timezone.utc)

        # Golden cross: short crosses above long
        if short_now > long_now and short_prev <= long_prev:
            logger.info(
                "BUY signal for %s | short_ma=%.4f long_ma=%.4f", symbol, short_now, long_now
            )
            return Signal(
                action="BUY",
                symbol=symbol,
                short_ma=short_now,
                long_ma=long_now,
                confidence=1.0,
                timestamp=timestamp,
            )

        # Death cross: short crosses below long
        if short_now < long_now and short_prev >= long_prev:
            logger.info(
                "SELL signal for %s | short_ma=%.4f long_ma=%.4f", symbol, short_now, long_now
            )
            return Signal(
                action="SELL",
                symbol=symbol,
                short_ma=short_now,
                long_ma=long_now,
                confidence=1.0,
                timestamp=timestamp,
            )

        logger.debug(
            "HOLD for %s | short_ma=%.4f long_ma=%.4f", symbol, short_now, long_now
        )
        return self._hold(symbol, short_now, long_now)

    def _hold(self, symbol: str, short_ma: float, long_ma: float) -> Signal:
        return Signal(
            action="HOLD",
            symbol=symbol,
            short_ma=short_ma,
            long_ma=long_ma,
            confidence=0.0,
            timestamp=datetime.now(tz=timezone.utc),
        )
