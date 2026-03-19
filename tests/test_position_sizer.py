from __future__ import annotations

import pytest

from src.broker.base import AccountInfo, PositionInfo
from src.risk.position_sizer import PositionSizer


def _account(equity=100_000.0, cash=100_000.0, buying_power=100_000.0, portfolio_value=100_000.0):
    return AccountInfo(
        equity=equity,
        cash=cash,
        buying_power=buying_power,
        portfolio_value=portfolio_value,
    )


def _position(symbol="AAPL", qty=10, avg=150.0, market_value=1500.0, unrealized_pl=0.0):
    return PositionInfo(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg,
        market_value=market_value,
        unrealized_pl=unrealized_pl,
    )


@pytest.fixture
def sizer(sample_config):
    return PositionSizer(sample_config)


def test_buy_calculates_correct_qty(sizer):
    # equity=100k, max_position_pct=0.10 → $10k alloc, price=$100 → qty=100
    result = sizer.calculate_order("AAPL", "buy", 100.0, _account(), [])
    assert result is not None
    assert result.qty == 100
    assert result.side == "buy"
    assert result.symbol == "AAPL"


def test_buy_no_pyramiding(sizer):
    # Already holding AAPL → should return None
    positions = [_position(symbol="AAPL", qty=10)]
    result = sizer.calculate_order("AAPL", "buy", 100.0, _account(), positions)
    assert result is None


def test_sell_returns_full_position_qty(sizer):
    positions = [_position(symbol="AAPL", qty=25)]
    result = sizer.calculate_order("AAPL", "sell", 100.0, _account(), positions)
    assert result is not None
    assert result.qty == 25
    assert result.side == "sell"


def test_sell_no_position_returns_none(sizer):
    result = sizer.calculate_order("AAPL", "sell", 100.0, _account(), [])
    assert result is None


def test_total_exposure_limit_blocks_buy(sizer):
    # max_total_exposure_pct=0.50 → max $50k total
    # existing positions = $50k → at limit
    positions = [_position(symbol="MSFT", qty=1, market_value=50_000.0)]
    result = sizer.calculate_order("AAPL", "buy", 100.0, _account(), positions)
    assert result is None


def test_zero_equity_returns_none(sizer):
    result = sizer.calculate_order("AAPL", "buy", 100.0, _account(equity=0.0), [])
    assert result is None


def test_zero_price_returns_none(sizer):
    result = sizer.calculate_order("AAPL", "buy", 0.0, _account(), [])
    assert result is None


def test_qty_rounds_down(sizer):
    # equity=100k, 10% → $10k, price=$101 → floor(10000/101) = 99
    result = sizer.calculate_order("AAPL", "buy", 101.0, _account(), [])
    assert result is not None
    assert result.qty == 99


def test_exposure_remaining_caps_buy(sizer):
    # max total = 50k, already at 45k → only $5k left → floor(5000/100) = 50
    positions = [_position(symbol="MSFT", qty=1, market_value=45_000.0)]
    result = sizer.calculate_order("AAPL", "buy", 100.0, _account(), positions)
    assert result is not None
    assert result.qty == 50
