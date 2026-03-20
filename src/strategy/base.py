from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

import pandas as pd


@dataclass(frozen=True)
class Signal:
    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    short_ma: float
    long_ma: float
    confidence: float
    timestamp: datetime
    # Optional confirmation indicator values (None when not computed)
    rsi: Optional[float] = field(default=None)
    bb_upper: Optional[float] = field(default=None)
    bb_mid: Optional[float] = field(default=None)
    bb_lower: Optional[float] = field(default=None)
    macd: Optional[float] = field(default=None)
    macd_signal: Optional[float] = field(default=None)
    macd_hist: Optional[float] = field(default=None)


class Strategy(ABC):
    @abstractmethod
    def compute_signal(self, symbol: str, bars: pd.DataFrame) -> Signal:
        """
        Compute a trading signal from the provided OHLCV DataFrame.
        The DataFrame must have columns [open, high, low, close, volume]
        sorted ascending by index (oldest → newest).
        """
