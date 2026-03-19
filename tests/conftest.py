from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

import pandas as pd
import pytest

# Ensure environment variables are set before any imports that trigger load_dotenv
os.environ.setdefault("APCA_API_KEY_ID", "test_key")
os.environ.setdefault("APCA_API_SECRET_KEY", "test_secret")
os.environ.setdefault("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
os.environ.setdefault("SYMBOLS", "AAPL,MSFT")
os.environ.setdefault("SHORT_MA_PERIOD", "5")
os.environ.setdefault("LONG_MA_PERIOD", "10")
os.environ.setdefault("MAX_POSITION_PCT", "0.10")
os.environ.setdefault("MAX_TOTAL_EXPOSURE_PCT", "0.50")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("DB_PATH", ":memory:")


def make_bars(closes: List[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of closing prices."""
    n = len(closes)
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * n,
        },
        index=index,
    )


@pytest.fixture
def sample_config():
    from src.config import load_config
    return load_config()
