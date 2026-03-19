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
    )
