from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns [open, high, low, close, volume]
        indexed by UTC timestamp, sorted ascending, for the most recent `limit` bars.
        """

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent closing price for the given symbol."""
