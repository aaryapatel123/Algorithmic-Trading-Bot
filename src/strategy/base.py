from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class Signal:
    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    short_ma: float
    long_ma: float
    confidence: float
    timestamp: datetime


class Strategy(ABC):
    @abstractmethod
    def compute_signal(self, symbol: str, bars: pd.DataFrame) -> Signal:
        """
        Compute a trading signal from the provided OHLCV DataFrame.
        The DataFrame must have columns [open, high, low, close, volume]
        sorted ascending by index (oldest → newest).
        """
