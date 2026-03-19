from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.broker.base import AccountInfo, ClockInfo, OrderResult, PositionInfo


def _make_mock_account(equity=100000.0, cash=50000.0, buying_power=50000.0, portfolio_value=100000.0):
    a = MagicMock()
    a.equity = str(equity)
    a.cash = str(cash)
    a.buying_power = str(buying_power)
    a.portfolio_value = str(portfolio_value)
    return a


def _make_mock_position(symbol="AAPL", qty=10, avg=150.0, market_value=1500.0, unrealized_pl=50.0):
    p = MagicMock()
    p.symbol = symbol
    p.qty = str(qty)
    p.avg_entry_price = str(avg)
    p.market_value = str(market_value)
    p.unrealized_pl = str(unrealized_pl)
    return p


def _make_mock_order(
    order_id="order-123",
    symbol="AAPL",
    qty=10,
    side="buy",
    status="accepted",
    submitted_at="2024-01-15T10:00:00+00:00",
):
    o = MagicMock()
    o.id = order_id
    o.symbol = symbol
    o.qty = str(qty)
    o.side = side
    o.status = status
    o.submitted_at = submitted_at
    return o


def _make_mock_clock(is_open=True):
    c = MagicMock()
    c.is_open = is_open
    c.next_open = "2024-01-16T09:30:00+00:00"
    c.next_close = "2024-01-15T16:00:00+00:00"
    return c


@pytest.fixture
def broker():
    from src.broker.alpaca_client import AlpacaBroker
    mock_rest = MagicMock()
    b = AlpacaBroker.__new__(AlpacaBroker)
    b._api = mock_rest
    return b, mock_rest


def test_get_account_returns_account_info(broker):
    b, rest = broker
    rest.get_account.return_value = _make_mock_account()
    account = b.get_account()
    assert isinstance(account, AccountInfo)
    assert account.equity == 100000.0
    assert account.cash == 50000.0


def test_get_positions_returns_list(broker):
    b, rest = broker
    rest.list_positions.return_value = [_make_mock_position()]
    positions = b.get_positions()
    assert len(positions) == 1
    assert isinstance(positions[0], PositionInfo)
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10


def test_get_position_not_found_returns_none(broker):
    b, rest = broker
    rest.get_position.side_effect = Exception("position does not exist")
    result = b.get_position("AAPL")
    assert result is None


def test_submit_market_order_success(broker):
    b, rest = broker
    rest.submit_order.return_value = _make_mock_order()
    result = b.submit_market_order("AAPL", 10, "buy")
    assert isinstance(result, OrderResult)
    assert result.order_id == "order-123"
    assert result.symbol == "AAPL"
    assert result.qty == 10
    assert result.side == "buy"


def test_submit_market_order_sends_correct_params(broker):
    b, rest = broker
    rest.submit_order.return_value = _make_mock_order()
    b.submit_market_order("AAPL", 5, "sell")
    rest.submit_order.assert_called_once_with(
        symbol="AAPL",
        qty="5",
        side="sell",
        type="market",
        time_in_force="day",
    )


def test_submit_market_order_raises_on_failure(broker):
    b, rest = broker
    rest.submit_order.side_effect = Exception("Insufficient funds")
    with pytest.raises(Exception, match="Insufficient funds"):
        b.submit_market_order("AAPL", 1000, "buy")


def test_is_market_open_true(broker):
    b, rest = broker
    rest.get_clock.return_value = _make_mock_clock(is_open=True)
    assert b.is_market_open() is True


def test_is_market_open_false(broker):
    b, rest = broker
    rest.get_clock.return_value = _make_mock_clock(is_open=False)
    assert b.is_market_open() is False


def test_get_clock_returns_clock_info(broker):
    b, rest = broker
    rest.get_clock.return_value = _make_mock_clock()
    clock = b.get_clock()
    assert isinstance(clock, ClockInfo)
    assert clock.is_open is True
