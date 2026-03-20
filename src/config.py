from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    api_key_id: str
    api_secret_key: str
    base_url: str
    symbols: List[str]
    short_ma_period: int
    long_ma_period: int
    bar_timeframe: str
    max_position_pct: float
    max_total_exposure_pct: float
    log_level: str
    db_path: str
    # Confirmation indicator params
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bb_period: int = 20
    bb_std_dev: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal_period: int = 9
    min_confirmations: int = 1


def load_config() -> Config:
    api_key_id = os.environ.get("APCA_API_KEY_ID", "")
    api_secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
    base_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key_id:
        raise ValueError("APCA_API_KEY_ID environment variable is required")
    if not api_secret_key:
        raise ValueError("APCA_API_SECRET_KEY environment variable is required")

    symbols_raw = os.environ.get("SYMBOLS", "")
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    if not symbols:
        raise ValueError("SYMBOLS environment variable must contain at least one symbol")

    short_ma_period = int(os.environ.get("SHORT_MA_PERIOD", "20"))
    long_ma_period = int(os.environ.get("LONG_MA_PERIOD", "50"))

    if short_ma_period <= 0:
        raise ValueError("SHORT_MA_PERIOD must be positive")
    if long_ma_period <= 0:
        raise ValueError("LONG_MA_PERIOD must be positive")
    if short_ma_period >= long_ma_period:
        raise ValueError("SHORT_MA_PERIOD must be less than LONG_MA_PERIOD")

    max_position_pct = float(os.environ.get("MAX_POSITION_PCT", "0.10"))
    max_total_exposure_pct = float(os.environ.get("MAX_TOTAL_EXPOSURE_PCT", "0.50"))

    if not (0 < max_position_pct <= 1.0):
        raise ValueError("MAX_POSITION_PCT must be between 0 and 1")
    if not (0 < max_total_exposure_pct <= 1.0):
        raise ValueError("MAX_TOTAL_EXPOSURE_PCT must be between 0 and 1")

    rsi_period = int(os.environ.get("RSI_PERIOD", "14"))
    rsi_overbought = float(os.environ.get("RSI_OVERBOUGHT", "70.0"))
    rsi_oversold = float(os.environ.get("RSI_OVERSOLD", "30.0"))
    bb_period = int(os.environ.get("BB_PERIOD", "20"))
    bb_std_dev = float(os.environ.get("BB_STD_DEV", "2.0"))
    macd_fast = int(os.environ.get("MACD_FAST", "12"))
    macd_slow = int(os.environ.get("MACD_SLOW", "26"))
    macd_signal_period = int(os.environ.get("MACD_SIGNAL", "9"))
    min_confirmations = int(os.environ.get("MIN_CONFIRMATIONS", "1"))

    return Config(
        api_key_id=api_key_id,
        api_secret_key=api_secret_key,
        base_url=base_url,
        symbols=symbols,
        short_ma_period=short_ma_period,
        long_ma_period=long_ma_period,
        bar_timeframe=os.environ.get("BAR_TIMEFRAME", "1D"),
        max_position_pct=max_position_pct,
        max_total_exposure_pct=max_total_exposure_pct,
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        db_path=os.environ.get("DB_PATH", "trading_bot.db"),
        rsi_period=rsi_period,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        bb_period=bb_period,
        bb_std_dev=bb_std_dev,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal_period=macd_signal_period,
        min_confirmations=min_confirmations,
    )
