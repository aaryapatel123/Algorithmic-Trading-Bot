from __future__ import annotations

import pytest
from tests.conftest import make_bars
from src.strategy.ma_crossover import MACrossoverStrategy


@pytest.fixture
def strategy():
    return MACrossoverStrategy(short_period=5, long_period=10)


def _golden_cross_prices() -> list:
    """
    Prices that produce a golden cross on the final bar.
    14 bars flat at 100, then a spike to 200.
    short_ma[-1]=140 > long_ma[-1]=110; short_ma[-2]=100 <= long_ma[-2]=100.
    """
    return [100.0] * 14 + [200.0]


def _death_cross_prices() -> list:
    """
    Prices that produce a death cross on the final bar.
    14 bars flat at 200, then a drop to 50.
    short_ma[-1]=170 < long_ma[-1]=185; short_ma[-2]=200 >= long_ma[-2]=200.
    """
    return [200.0] * 14 + [50.0]


def test_buy_signal_on_golden_cross(strategy):
    bars = make_bars(_golden_cross_prices())
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "BUY"
    assert signal.symbol == "AAPL"
    assert signal.short_ma > signal.long_ma


def test_sell_signal_on_death_cross(strategy):
    bars = make_bars(_death_cross_prices())
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "SELL"
    assert signal.symbol == "AAPL"
    assert signal.short_ma < signal.long_ma


def test_hold_when_no_crossover(strategy):
    # Flat prices → no crossover
    bars = make_bars([100.0] * 20)
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "HOLD"


def test_hold_with_insufficient_data(strategy):
    # Only 10 bars but long_period=10, need at least 11
    bars = make_bars([100.0] * 10)
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "HOLD"


def test_hold_with_empty_bars(strategy):
    import pandas as pd
    bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "HOLD"


def test_signal_is_immutable(strategy):
    bars = make_bars([100.0] * 20)
    signal = strategy.compute_signal("AAPL", bars)
    with pytest.raises((AttributeError, TypeError)):
        signal.action = "BUY"


def test_signal_has_timestamp(strategy):
    from datetime import timezone
    bars = make_bars([100.0] * 20)
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.timestamp is not None
    assert signal.timestamp.tzinfo == timezone.utc


def test_invalid_short_period():
    with pytest.raises(ValueError, match="short_period"):
        MACrossoverStrategy(short_period=0, long_period=10)


def test_invalid_long_period():
    with pytest.raises(ValueError, match="long_period"):
        MACrossoverStrategy(short_period=5, long_period=0)


def test_short_gte_long_raises():
    with pytest.raises(ValueError, match="short_period"):
        MACrossoverStrategy(short_period=10, long_period=5)


def test_equal_periods_raises():
    with pytest.raises(ValueError, match="short_period"):
        MACrossoverStrategy(short_period=10, long_period=10)
