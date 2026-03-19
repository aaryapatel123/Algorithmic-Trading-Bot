from __future__ import annotations

from datetime import datetime, timezone

import pytest
from peewee import SqliteDatabase

from src.models.trade import (
    BotState,
    SignalRecord,
    TradeRecord,
    _db,
    get_state,
    init_db,
    set_state,
)


@pytest.fixture(autouse=True)
def in_memory_db():
    """Use an in-memory SQLite database for each test."""
    _db.init(":memory:")
    _db.connect(reuse_if_open=True)
    _db.create_tables([SignalRecord, TradeRecord, BotState], safe=True)
    yield
    _db.drop_tables([TradeRecord, SignalRecord, BotState])
    _db.close()


def _create_signal():
    return SignalRecord.create(
        symbol="AAPL",
        action="BUY",
        short_ma=21.5,
        long_ma=50.3,
        confidence=1.0,
        signal_timestamp=datetime.now(tz=timezone.utc),
    )


def test_create_signal_record():
    record = _create_signal()
    assert record.id is not None
    assert record.symbol == "AAPL"
    assert record.action == "BUY"


def test_query_signal_record():
    _create_signal()
    records = list(SignalRecord.select())
    assert len(records) == 1
    assert records[0].symbol == "AAPL"


def test_create_trade_record():
    signal = _create_signal()
    trade = TradeRecord.create(
        symbol="AAPL",
        side="buy",
        qty=100,
        order_id="order-abc",
        status="accepted",
        signal=signal,
        trade_timestamp=datetime.now(tz=timezone.utc),
    )
    assert trade.id is not None
    assert trade.symbol == "AAPL"
    assert trade.qty == 100


def test_trade_foreign_key_to_signal():
    signal = _create_signal()
    trade = TradeRecord.create(
        symbol="AAPL",
        side="buy",
        qty=10,
        order_id="ord-1",
        status="accepted",
        signal=signal,
        trade_timestamp=datetime.now(tz=timezone.utc),
    )
    retrieved = TradeRecord.get_by_id(trade.id)
    assert retrieved.signal_id == signal.id


def test_bot_state_set_and_get():
    set_state("last_run", "2024-01-15")
    assert get_state("last_run") == "2024-01-15"


def test_bot_state_get_missing_key_returns_default():
    assert get_state("nonexistent", "fallback") == "fallback"


def test_bot_state_upsert_updates_value():
    set_state("key1", "value1")
    set_state("key1", "value2")
    assert get_state("key1") == "value2"
    assert BotState.select().count() == 1
