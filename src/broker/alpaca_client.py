from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Literal, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.broker.base import AccountInfo, Broker, ClockInfo, OrderResult, PositionInfo
from src.config import Config

logger = logging.getLogger(__name__)


class AlpacaBroker(Broker):
    def __init__(self, config: Config) -> None:
        self._client = TradingClient(
            api_key=config.api_key_id,
            secret_key=config.api_secret_key,
            paper=True,
            url_override=config.base_url,
        )
        logger.info("Alpaca broker initialized (base_url=%s)", config.base_url)

    def get_account(self) -> AccountInfo:
        account = self._client.get_account()
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
        )

    def get_positions(self) -> List[PositionInfo]:
        positions = self._client.get_all_positions()
        return [self._map_position(p) for p in positions]

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        try:
            pos = self._client.get_open_position(symbol)
            return self._map_position(pos)
        except Exception as exc:
            if "position does not exist" in str(exc).lower() or "404" in str(exc):
                return None
            raise

    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: Literal["buy", "sell"],
    ) -> OrderResult:
        logger.info("Submitting %s order: %s x%d", side.upper(), symbol, qty)
        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(order_request)
            result = OrderResult(
                order_id=str(order.id),
                symbol=order.symbol,
                qty=int(order.qty),
                side=order.side.value,
                status=order.status.value,
                submitted_at=self._parse_dt(order.submitted_at),
            )
            logger.info(
                "Order submitted: id=%s status=%s", result.order_id, result.status
            )
            return result
        except Exception as exc:
            logger.error("Order submission failed for %s: %s", symbol, exc)
            raise

    def is_market_open(self) -> bool:
        return self.get_clock().is_open

    def get_clock(self) -> ClockInfo:
        clock = self._client.get_clock()
        return ClockInfo(
            is_open=clock.is_open,
            next_open=self._parse_dt(clock.next_open),
            next_close=self._parse_dt(clock.next_close),
        )

    @staticmethod
    def _map_position(pos) -> PositionInfo:
        return PositionInfo(
            symbol=pos.symbol,
            qty=int(pos.qty),
            avg_entry_price=float(pos.avg_entry_price),
            market_value=float(pos.market_value),
            unrealized_pl=float(pos.unrealized_pl),
        )

    @staticmethod
    def _parse_dt(value) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt
