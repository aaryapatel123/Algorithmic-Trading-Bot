from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.broker.base import AccountInfo, ClockInfo, OrderResult, PositionInfo
from src.models.trade import _db, BotState, SignalRecord, TradeRecord, init_db
from src.strategy.base import Signal
from tests.conftest import make_bars


@pytest.fixture(autouse=True)
def in_memory_db():
    _db.init(":memory:")
    _db.connect(reuse_if_open=True)
    _db.create_tables([SignalRecord, TradeRecord, BotState], safe=True)
    yield
    _db.drop_tables([TradeRecord, SignalRecord, BotState])
    _db.close()


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_account.return_value = AccountInfo(
        equity=100_000.0, cash=100_000.0, buying_power=100_000.0, portfolio_value=100_000.0
    )
    broker.get_positions.return_value = []
    broker.is_market_open.return_value = True
    broker.get_clock.return_value = ClockInfo(
        is_open=True,
        next_open=datetime(2024, 1, 16, 9, 30, tzinfo=timezone.utc),
        next_close=datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc),
    )
    broker.submit_market_order.return_value = OrderResult(
        order_id="test-order-1",
        symbol="AAPL",
        qty=100,
        side="buy",
        status="accepted",
        submitted_at=datetime.now(tz=timezone.utc),
    )
    return broker


@pytest.fixture
def mock_data_provider():
    provider = MagicMock()
    provider.get_historical_bars.return_value = make_bars(
        [100.0] * 14 + [200.0]  # golden cross on final bar
    )
    provider.get_latest_price.return_value = 125.0
    return provider


def _build_bot(sample_config, mock_broker, mock_data_provider):
    from src.main import TradingBot
    bot = TradingBot.__new__(TradingBot)
    bot._config = sample_config
    bot._broker = mock_broker
    bot._data = mock_data_provider
    from src.strategy.ma_crossover import MACrossoverStrategy
    from src.risk.position_sizer import PositionSizer
    bot._strategy = MACrossoverStrategy(
        short_period=sample_config.short_ma_period,
        long_period=sample_config.long_ma_period,
    )
    bot._sizer = PositionSizer(sample_config)
    bot._running = True
    return bot


def test_buy_signal_submits_order(sample_config, mock_broker, mock_data_provider):
    bot = _build_bot(sample_config, mock_broker, mock_data_provider)
    # Compute a BUY signal manually and verify order is placed
    bars = mock_data_provider.get_historical_bars.return_value
    signal = bot._strategy.compute_signal("AAPL", bars)
    assert signal.action == "BUY"

    bot._process_symbol("AAPL", mock_broker.get_account(), mock_broker.get_positions())
    mock_broker.submit_market_order.assert_called_once()


def test_buy_signal_persists_signal_and_trade(sample_config, mock_broker, mock_data_provider):
    bot = _build_bot(sample_config, mock_broker, mock_data_provider)
    bot._process_symbol("AAPL", mock_broker.get_account(), mock_broker.get_positions())

    assert SignalRecord.select().count() == 1
    assert TradeRecord.select().count() == 1


def test_hold_signal_no_order(sample_config, mock_broker, mock_data_provider):
    # Return flat prices so strategy returns HOLD
    mock_data_provider.get_historical_bars.return_value = make_bars([100.0] * 20)
    bot = _build_bot(sample_config, mock_broker, mock_data_provider)
    bot._process_symbol("AAPL", mock_broker.get_account(), mock_broker.get_positions())
    mock_broker.submit_market_order.assert_not_called()
    assert TradeRecord.select().count() == 0


def test_sell_signal_with_no_position_no_order(sample_config, mock_broker, mock_data_provider):
    # Death cross prices
    mock_data_provider.get_historical_bars.return_value = make_bars(
        [200.0] * 14 + [50.0]  # death cross on final bar
    )
    bot = _build_bot(sample_config, mock_broker, mock_data_provider)
    mock_broker.get_positions.return_value = []
    bot._process_symbol("AAPL", mock_broker.get_account(), mock_broker.get_positions())
    mock_broker.submit_market_order.assert_not_called()


def test_duplicate_signal_not_resubmitted(sample_config, mock_broker, mock_data_provider):
    bot = _build_bot(sample_config, mock_broker, mock_data_provider)
    # First call places order
    bot._process_symbol("AAPL", mock_broker.get_account(), mock_broker.get_positions())
    # Second call with same data — same signal timestamp → duplicate, no second order
    bot._process_symbol("AAPL", mock_broker.get_account(), mock_broker.get_positions())
    assert mock_broker.submit_market_order.call_count == 1
