from __future__ import annotations

"""
Multi-signal strategy: MA crossover primary trigger + RSI, Bollinger Bands,
and MACD as confirmation filters.

Confirmation logic
------------------
BUY (golden cross):
  - RSI < rsi_overbought            → not overbought, room to run
  - close <= bb_mid                 → price below midband, still has upside
  - macd_line > macd_signal_line    → bullish MACD momentum

SELL (death cross):
  - RSI > rsi_oversold              → not oversold, room to fall
  - close >= bb_mid                 → price above midband, still has downside
  - macd_line < macd_signal_line    → bearish MACD momentum

confidence  = confirmed_count / 3.0
If confirmed_count < min_confirmations the crossover is downgraded to HOLD.
"""

import logging
import math
from datetime import datetime, timezone

import pandas as pd

from src.strategy.base import Signal, Strategy
from src.strategy.indicators import (
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
)

logger = logging.getLogger(__name__)


class MultiSignalStrategy(Strategy):
    def __init__(
        self,
        short_period: int,
        long_period: int,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_period: int = 9,
        min_confirmations: int = 1,
    ) -> None:
        if short_period <= 0:
            raise ValueError("short_period must be positive")
        if long_period <= 0:
            raise ValueError("long_period must be positive")
        if short_period >= long_period:
            raise ValueError("short_period must be less than long_period")
        if not (0 < rsi_overbought <= 100):
            raise ValueError("rsi_overbought must be in (0, 100]")
        if not (0 <= rsi_oversold < 100):
            raise ValueError("rsi_oversold must be in [0, 100)")
        if rsi_oversold >= rsi_overbought:
            raise ValueError("rsi_oversold must be less than rsi_overbought")
        if not (0 <= min_confirmations <= 3):
            raise ValueError("min_confirmations must be between 0 and 3")

        self._short_period = short_period
        self._long_period = long_period
        self._rsi_period = rsi_period
        self._rsi_overbought = rsi_overbought
        self._rsi_oversold = rsi_oversold
        self._bb_period = bb_period
        self._bb_std_dev = bb_std_dev
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal_period = macd_signal_period
        self._min_confirmations = min_confirmations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_signal(self, symbol: str, bars: pd.DataFrame) -> Signal:
        if bars.empty or "close" not in bars.columns:
            logger.warning("Empty or invalid bars for %s — returning HOLD", symbol)
            return self._hold(symbol, 0.0, 0.0)

        close = bars["close"]
        min_bars = max(
            self._long_period + 1,
            self._bb_period,
            self._macd_slow + self._macd_signal_period,
            self._rsi_period + 1,
        )
        if len(close) < min_bars:
            logger.warning(
                "Insufficient data for %s (%d bars, need %d) — returning HOLD",
                symbol,
                len(close),
                min_bars,
            )
            return self._hold(symbol, 0.0, 0.0)

        # ── Moving averages ─────────────────────────────────────────────
        short_ma = close.rolling(self._short_period).mean()
        long_ma = close.rolling(self._long_period).mean()

        valid = short_ma.notna() & long_ma.notna()
        if valid.sum() < 2:
            logger.warning("Not enough valid MA values for %s — returning HOLD", symbol)
            return self._hold(symbol, float(short_ma.iloc[-1]), float(long_ma.iloc[-1]))

        short_now = float(short_ma.iloc[-1])
        short_prev = float(short_ma[valid].iloc[-2])
        long_now = float(long_ma.iloc[-1])
        long_prev = float(long_ma[valid].iloc[-2])

        # ── Confirmation indicators ──────────────────────────────────────
        rsi = compute_rsi(close, self._rsi_period)
        bb_upper, bb_mid, bb_lower = compute_bollinger_bands(
            close, self._bb_period, self._bb_std_dev
        )
        macd_line, signal_line, histogram = compute_macd(
            close, self._macd_fast, self._macd_slow, self._macd_signal_period
        )

        rsi_now = float(rsi.iloc[-1])
        bb_upper_now = float(bb_upper.iloc[-1])
        bb_mid_now = float(bb_mid.iloc[-1])
        bb_lower_now = float(bb_lower.iloc[-1])
        macd_now = float(macd_line.iloc[-1])
        macd_signal_now = float(signal_line.iloc[-1])
        macd_hist_now = float(histogram.iloc[-1])
        close_now = float(close.iloc[-1])

        # ── Detect crossover ────────────────────────────────────────────
        golden_cross = short_now > long_now and short_prev <= long_prev
        death_cross = short_now < long_now and short_prev >= long_prev

        if not golden_cross and not death_cross:
            logger.debug(
                "HOLD for %s | short_ma=%.4f long_ma=%.4f rsi=%.2f",
                symbol,
                short_now,
                long_now,
                rsi_now,
            )
            return self._make_signal(
                "HOLD",
                symbol,
                short_now,
                long_now,
                0.0,
                rsi_now,
                bb_upper_now,
                bb_mid_now,
                bb_lower_now,
                macd_now,
                macd_signal_now,
                macd_hist_now,
            )

        # ── Count confirmations ─────────────────────────────────────────
        if golden_cross:
            confirmations = self._count_buy_confirmations(
                rsi_now, close_now, bb_mid_now, macd_now, macd_signal_now
            )
            if confirmations < self._min_confirmations:
                logger.info(
                    "Golden cross for %s but only %d/%d confirmations — HOLD",
                    symbol,
                    confirmations,
                    self._min_confirmations,
                )
                return self._make_signal(
                    "HOLD",
                    symbol,
                    short_now,
                    long_now,
                    0.0,
                    rsi_now,
                    bb_upper_now,
                    bb_mid_now,
                    bb_lower_now,
                    macd_now,
                    macd_signal_now,
                    macd_hist_now,
                )
            confidence = confirmations / 3.0
            logger.info(
                "BUY signal for %s | short_ma=%.4f long_ma=%.4f "
                "rsi=%.2f macd=%.4f conf=%.2f",
                symbol,
                short_now,
                long_now,
                rsi_now,
                macd_now,
                confidence,
            )
            return self._make_signal(
                "BUY",
                symbol,
                short_now,
                long_now,
                confidence,
                rsi_now,
                bb_upper_now,
                bb_mid_now,
                bb_lower_now,
                macd_now,
                macd_signal_now,
                macd_hist_now,
            )

        # death_cross
        confirmations = self._count_sell_confirmations(
            rsi_now, close_now, bb_mid_now, macd_now, macd_signal_now
        )
        if confirmations < self._min_confirmations:
            logger.info(
                "Death cross for %s but only %d/%d confirmations — HOLD",
                symbol,
                confirmations,
                self._min_confirmations,
            )
            return self._make_signal(
                "HOLD",
                symbol,
                short_now,
                long_now,
                0.0,
                rsi_now,
                bb_upper_now,
                bb_mid_now,
                bb_lower_now,
                macd_now,
                macd_signal_now,
                macd_hist_now,
            )
        confidence = confirmations / 3.0
        logger.info(
            "SELL signal for %s | short_ma=%.4f long_ma=%.4f "
            "rsi=%.2f macd=%.4f conf=%.2f",
            symbol,
            short_now,
            long_now,
            rsi_now,
            macd_now,
            confidence,
        )
        return self._make_signal(
            "SELL",
            symbol,
            short_now,
            long_now,
            confidence,
            rsi_now,
            bb_upper_now,
            bb_mid_now,
            bb_lower_now,
            macd_now,
            macd_signal_now,
            macd_hist_now,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_buy_confirmations(
        self,
        rsi: float,
        close: float,
        bb_mid: float,
        macd: float,
        macd_sig: float,
    ) -> int:
        count = 0
        if not math.isnan(rsi) and rsi < self._rsi_overbought:
            count += 1
        if not math.isnan(bb_mid) and close <= bb_mid:
            count += 1
        if not math.isnan(macd) and not math.isnan(macd_sig) and macd > macd_sig:
            count += 1
        return count

    def _count_sell_confirmations(
        self,
        rsi: float,
        close: float,
        bb_mid: float,
        macd: float,
        macd_sig: float,
    ) -> int:
        count = 0
        if not math.isnan(rsi) and rsi > self._rsi_oversold:
            count += 1
        if not math.isnan(bb_mid) and close >= bb_mid:
            count += 1
        if not math.isnan(macd) and not math.isnan(macd_sig) and macd < macd_sig:
            count += 1
        return count

    def _hold(self, symbol: str, short_ma: float, long_ma: float) -> Signal:
        return Signal(
            action="HOLD",
            symbol=symbol,
            short_ma=short_ma,
            long_ma=long_ma,
            confidence=0.0,
            timestamp=datetime.now(tz=timezone.utc),
        )

    def _make_signal(
        self,
        action: str,
        symbol: str,
        short_ma: float,
        long_ma: float,
        confidence: float,
        rsi: float,
        bb_upper: float,
        bb_mid: float,
        bb_lower: float,
        macd: float,
        macd_signal: float,
        macd_hist: float,
    ) -> Signal:
        return Signal(
            action=action,  # type: ignore[arg-type]
            symbol=symbol,
            short_ma=short_ma,
            long_ma=long_ma,
            confidence=confidence,
            timestamp=datetime.now(tz=timezone.utc),
            rsi=rsi,
            bb_upper=bb_upper,
            bb_mid=bb_mid,
            bb_lower=bb_lower,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
        )
