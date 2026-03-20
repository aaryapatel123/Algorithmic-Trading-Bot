from __future__ import annotations

import pandas as pd
import pytest
from tests.conftest import make_bars
from src.strategy.multi_signal import MultiSignalStrategy


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def strategy():
    """Strategy with min_confirmations=0 so MA crossover alone is enough."""
    return MultiSignalStrategy(short_period=5, long_period=10, min_confirmations=0)


@pytest.fixture
def strict_strategy():
    """Strategy requiring all 3 confirmations."""
    return MultiSignalStrategy(short_period=5, long_period=10, min_confirmations=3)


# ──────────────────────────────────────────────────────────────────────────────
# Price helpers  (same as ma_crossover tests — crossover on final bar)
# ──────────────────────────────────────────────────────────────────────────────

def _golden_cross_prices(n: int = 60) -> list[float]:
    """Flat run followed by a big spike to force a golden cross on the last bar."""
    return [100.0] * (n - 1) + [200.0]


def _death_cross_prices(n: int = 60) -> list[float]:
    """Flat run followed by a big drop to force a death cross on the last bar."""
    return [200.0] * (n - 1) + [50.0]


# ──────────────────────────────────────────────────────────────────────────────
# Basic crossover behaviour (min_confirmations=0)
# ──────────────────────────────────────────────────────────────────────────────

def test_buy_on_golden_cross_no_min_conf(strategy):
    bars = make_bars(_golden_cross_prices())
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "BUY"
    assert signal.symbol == "AAPL"
    assert signal.short_ma > signal.long_ma


def test_sell_on_death_cross_no_min_conf(strategy):
    bars = make_bars(_death_cross_prices())
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "SELL"
    assert signal.symbol == "AAPL"
    assert signal.short_ma < signal.long_ma


def test_hold_when_no_crossover(strategy):
    bars = make_bars([100.0] * 60)
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "HOLD"


def test_hold_with_insufficient_data(strategy):
    bars = make_bars([100.0] * 10)
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "HOLD"


def test_hold_with_empty_bars(strategy):
    bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "HOLD"


# ──────────────────────────────────────────────────────────────────────────────
# Indicator values are populated
# ──────────────────────────────────────────────────────────────────────────────

def test_signal_has_indicator_values(strategy):
    bars = make_bars([100.0] * 60)
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.rsi is not None
    assert signal.bb_upper is not None
    assert signal.bb_mid is not None
    assert signal.bb_lower is not None
    assert signal.macd is not None
    assert signal.macd_signal is not None
    assert signal.macd_hist is not None


def test_buy_signal_includes_indicator_values(strategy):
    bars = make_bars(_golden_cross_prices())
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.action == "BUY"
    assert signal.rsi is not None
    import math
    assert not math.isnan(signal.rsi)
    assert 0 <= signal.rsi <= 100
    assert signal.bb_upper > signal.bb_mid > signal.bb_lower


# ──────────────────────────────────────────────────────────────────────────────
# Confidence scoring
# ──────────────────────────────────────────────────────────────────────────────

def test_confidence_between_0_and_1(strategy):
    bars = make_bars(_golden_cross_prices())
    signal = strategy.compute_signal("AAPL", bars)
    assert 0.0 <= signal.confidence <= 1.0


def test_hold_when_confidence_below_min_confirmations(strict_strategy):
    """strict_strategy needs 3/3 confirmations — flat prices won't provide them."""
    bars = make_bars(_golden_cross_prices())
    # With flat-then-spike prices RSI will be near 100 (all gains) → overbought
    # so RSI confirmation fails → at most 2/3 → HOLD with min_confirmations=3
    signal = strict_strategy.compute_signal("AAPL", bars)
    # Either no crossover detected, or confirmations < 3 → HOLD in both cases
    assert signal.action == "HOLD"


# ──────────────────────────────────────────────────────────────────────────────
# Signal immutability
# ──────────────────────────────────────────────────────────────────────────────

def test_signal_is_immutable(strategy):
    bars = make_bars([100.0] * 60)
    signal = strategy.compute_signal("AAPL", bars)
    with pytest.raises((AttributeError, TypeError)):
        signal.action = "BUY"  # type: ignore[misc]


def test_signal_has_utc_timestamp(strategy):
    from datetime import timezone
    bars = make_bars([100.0] * 60)
    signal = strategy.compute_signal("AAPL", bars)
    assert signal.timestamp is not None
    assert signal.timestamp.tzinfo == timezone.utc


# ──────────────────────────────────────────────────────────────────────────────
# Constructor validation
# ──────────────────────────────────────────────────────────────────────────────

def test_invalid_short_period():
    with pytest.raises(ValueError, match="short_period"):
        MultiSignalStrategy(short_period=0, long_period=10)


def test_short_gte_long_raises():
    with pytest.raises(ValueError, match="short_period"):
        MultiSignalStrategy(short_period=10, long_period=5)


def test_invalid_rsi_overbought():
    with pytest.raises(ValueError, match="rsi_overbought"):
        MultiSignalStrategy(short_period=5, long_period=10, rsi_overbought=0)


def test_invalid_rsi_levels():
    with pytest.raises(ValueError, match="rsi_oversold"):
        MultiSignalStrategy(short_period=5, long_period=10, rsi_oversold=80, rsi_overbought=70)


def test_invalid_min_confirmations():
    with pytest.raises(ValueError, match="min_confirmations"):
        MultiSignalStrategy(short_period=5, long_period=10, min_confirmations=4)
