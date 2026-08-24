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
    symbols: List[str]          # universe override; empty = fetch S&P 500 dynamically
    top_n: int                  # number of top momentum stocks to hold (default 3)
    momentum_days: int          # 12-month lookback in trading days (default 252)
    ma_period: int              # SPY MA period for crash filter (default 200)
    max_position_pct: float     # max single-position % of equity (safety guard)
    max_total_exposure_pct: float  # max total equity exposure (safety guard)
    log_level: str
    db_path: str


def load_config() -> Config:
    api_key_id = os.environ.get("ALPACA_API_KEY", "")
    api_secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key_id:
        raise ValueError("ALPACA_API_KEY environment variable is required")
    if not api_secret_key:
        raise ValueError("ALPACA_SECRET_KEY environment variable is required")

    # SYMBOLS is optional: if empty the bot fetches the S&P 500 universe at runtime
    symbols_raw = os.environ.get("SYMBOLS", "")
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    top_n = int(os.environ.get("TOP_N", "3"))
    momentum_days = int(os.environ.get("MOMENTUM_DAYS", "252"))
    ma_period = int(os.environ.get("MA_PERIOD", "200"))

    if top_n <= 0:
        raise ValueError("TOP_N must be positive")
    if momentum_days <= 0:
        raise ValueError("MOMENTUM_DAYS must be positive")
    if ma_period <= 0:
        raise ValueError("MA_PERIOD must be positive")

    max_position_pct = float(os.environ.get("MAX_POSITION_PCT", "0.40"))
    max_total_exposure_pct = float(os.environ.get("MAX_TOTAL_EXPOSURE_PCT", "0.95"))

    if not (0 < max_position_pct <= 1.0):
        raise ValueError("MAX_POSITION_PCT must be between 0 and 1")
    if not (0 < max_total_exposure_pct <= 1.0):
        raise ValueError("MAX_TOTAL_EXPOSURE_PCT must be between 0 and 1")

    return Config(
        api_key_id=api_key_id,
        api_secret_key=api_secret_key,
        base_url=base_url,
        symbols=symbols,
        top_n=top_n,
        momentum_days=momentum_days,
        ma_period=ma_period,
        max_position_pct=max_position_pct,
        max_total_exposure_pct=max_total_exposure_pct,
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        db_path=os.environ.get("DB_PATH", "trading_bot.db"),
    )
