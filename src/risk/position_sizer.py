from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Literal, Optional

from src.broker.base import AccountInfo, PositionInfo
from src.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    qty: int
    side: Literal["buy", "sell"]


class PositionSizer:
    def __init__(self, config: Config) -> None:
        self._config = config

    def calculate_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        current_price: float,
        account: AccountInfo,
        positions: List[PositionInfo],
    ) -> Optional[OrderRequest]:
        if account.equity <= 0:
            logger.warning("Account equity is zero or negative — skipping order")
            return None

        if side == "sell":
            return self._size_sell(symbol, positions)

        return self._size_buy(symbol, current_price, account, positions)

    def _size_sell(
        self,
        symbol: str,
        positions: List[PositionInfo],
    ) -> Optional[OrderRequest]:
        position = next((p for p in positions if p.symbol == symbol), None)
        if position is None or position.qty <= 0:
            logger.info("No position in %s to sell", symbol)
            return None
        return OrderRequest(symbol=symbol, qty=position.qty, side="sell")

    def _size_buy(
        self,
        symbol: str,
        current_price: float,
        account: AccountInfo,
        positions: List[PositionInfo],
    ) -> Optional[OrderRequest]:
        # No pyramiding: skip if we already hold this symbol
        if any(p.symbol == symbol and p.qty > 0 for p in positions):
            logger.info("Already holding %s — skipping BUY (no pyramiding)", symbol)
            return None

        if current_price <= 0:
            logger.warning("Invalid price %.4f for %s — skipping", current_price, symbol)
            return None

        # Check total current exposure
        current_exposure = sum(p.market_value for p in positions)
        max_total = account.equity * self._config.max_total_exposure_pct
        remaining_capacity = max_total - current_exposure

        if remaining_capacity <= 0:
            logger.info(
                "Total exposure limit reached (%.2f / %.2f) — skipping %s",
                current_exposure,
                max_total,
                symbol,
            )
            return None

        # Max dollar amount for this position
        max_position_dollars = account.equity * self._config.max_position_pct
        alloc_dollars = min(max_position_dollars, remaining_capacity)

        qty = math.floor(alloc_dollars / current_price)
        if qty <= 0:
            logger.info(
                "Calculated qty=0 for %s (price=%.2f, alloc=%.2f) — skipping",
                symbol,
                current_price,
                alloc_dollars,
            )
            return None

        logger.info(
            "BUY order sized: %s x%d @ %.2f (alloc $%.2f)",
            symbol,
            qty,
            current_price,
            alloc_dollars,
        )
        return OrderRequest(symbol=symbol, qty=qty, side="buy")
