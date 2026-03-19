from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional


@dataclass(frozen=True)
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    qty: int
    avg_entry_price: float
    market_value: float
    unrealized_pl: float


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    symbol: str
    qty: int
    side: Literal["buy", "sell"]
    status: str
    submitted_at: datetime


@dataclass(frozen=True)
class ClockInfo:
    is_open: bool
    next_open: datetime
    next_close: datetime


class Broker(ABC):
    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Return current account information."""

    @abstractmethod
    def get_positions(self) -> List[PositionInfo]:
        """Return all current open positions."""

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Return position for a specific symbol, or None if not held."""

    @abstractmethod
    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: Literal["buy", "sell"],
    ) -> OrderResult:
        """Submit a market order for the given symbol."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """Return True if the market is currently open for trading."""

    @abstractmethod
    def get_clock(self) -> ClockInfo:
        """Return current market clock information."""
