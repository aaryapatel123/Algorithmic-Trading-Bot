from __future__ import annotations

from datetime import datetime

from peewee import (
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

_db = SqliteDatabase(None)  # initialized via init_db()


class BaseModel(Model):
    class Meta:
        database = _db


class SignalRecord(BaseModel):
    symbol = CharField(max_length=10)
    action = CharField(max_length=4)  # BUY, SELL, HOLD
    short_ma = FloatField()
    long_ma = FloatField()
    confidence = FloatField()
    signal_timestamp = DateTimeField()
    created_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = "signals"


class TradeRecord(BaseModel):
    symbol = CharField(max_length=10)
    side = CharField(max_length=4)  # buy, sell
    qty = IntegerField()
    order_id = CharField(max_length=64)
    status = CharField(max_length=32)
    signal = ForeignKeyField(SignalRecord, backref="trades", null=True)
    trade_timestamp = DateTimeField()
    created_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = "trades"


class BotState(BaseModel):
    key = CharField(max_length=64, unique=True)
    value = TextField()
    updated_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = "bot_state"


def init_db(db_path: str) -> None:
    _db.init(db_path)
    _db.connect(reuse_if_open=True)
    _db.create_tables([SignalRecord, TradeRecord, BotState], safe=True)


def close_db() -> None:
    if not _db.is_closed():
        _db.close()


def set_state(key: str, value: str) -> None:
    BotState.insert(key=key, value=value, updated_at=datetime.utcnow()).on_conflict(
        conflict_target=[BotState.key],
        update={BotState.value: value, BotState.updated_at: datetime.utcnow()},
    ).execute()


def get_state(key: str, default: str = "") -> str:
    try:
        record = BotState.get(BotState.key == key)
        return record.value
    except BotState.DoesNotExist:
        return default
