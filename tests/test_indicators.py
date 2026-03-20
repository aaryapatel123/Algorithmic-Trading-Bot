from __future__ import annotations

import math

import pandas as pd
import pytest

from src.strategy.indicators import compute_bollinger_bands, compute_macd, compute_rsi


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeRSI:
    def test_returns_series_same_length(self):
        prices = _series(list(range(1, 21)))
        result = compute_rsi(prices, period=14)
        assert len(result) == len(prices)

    def test_first_values_are_nan(self):
        prices = _series(list(range(1, 21)))
        result = compute_rsi(prices, period=14)
        assert result.iloc[0:14].isna().all()

    def test_rsi_between_0_and_100(self):
        import numpy as np
        rng = np.random.default_rng(42)
        prices = _series((100 + rng.standard_normal(100).cumsum()).tolist())
        result = compute_rsi(prices, period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_all_gains_gives_rsi_100(self):
        # Strictly increasing prices → all gains, no losses → RSI → 100
        prices = _series([float(i) for i in range(1, 20)])
        result = compute_rsi(prices, period=5)
        assert math.isclose(result.iloc[-1], 100.0, abs_tol=1e-6)

    def test_all_losses_gives_rsi_0(self):
        # Strictly decreasing prices → all losses, no gains → RSI → 0
        prices = _series([float(20 - i) for i in range(20)])
        result = compute_rsi(prices, period=5)
        assert math.isclose(result.iloc[-1], 0.0, abs_tol=1e-6)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            compute_rsi(_series([1.0, 2.0]), period=0)


# ──────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeBollingerBands:
    def test_returns_three_series(self):
        prices = _series([100.0] * 30)
        upper, mid, lower = compute_bollinger_bands(prices, period=20)
        assert len(upper) == len(mid) == len(lower) == 30

    def test_flat_prices_zero_width(self):
        prices = _series([100.0] * 25)
        upper, mid, lower = compute_bollinger_bands(prices, period=20)
        # std of a constant series = 0 → upper == mid == lower
        assert math.isclose(float(upper.iloc[-1]), float(mid.iloc[-1]), abs_tol=1e-9)
        assert math.isclose(float(lower.iloc[-1]), float(mid.iloc[-1]), abs_tol=1e-9)

    def test_upper_gt_mid_gt_lower_for_varied_prices(self):
        import numpy as np
        rng = np.random.default_rng(0)
        prices = _series((100 + rng.standard_normal(50).cumsum()).tolist())
        upper, mid, lower = compute_bollinger_bands(prices, period=20)
        u, m, l = float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1])
        assert u > m > l

    def test_mid_is_rolling_mean(self):
        prices = _series([float(i) for i in range(1, 26)])
        _, mid, _ = compute_bollinger_bands(prices, period=20)
        expected = prices.rolling(20).mean()
        pd.testing.assert_series_equal(mid, expected)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            compute_bollinger_bands(_series([1.0, 2.0]), period=0)

    def test_invalid_std_dev_raises(self):
        with pytest.raises(ValueError):
            compute_bollinger_bands(_series([1.0] * 25), std_dev=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# MACD
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeMACD:
    def test_returns_three_series(self):
        prices = _series([float(i) for i in range(1, 50)])
        macd_line, signal_line, hist = compute_macd(prices, fast=12, slow=26, signal=9)
        assert len(macd_line) == len(signal_line) == len(hist) == 49

    def test_histogram_equals_macd_minus_signal(self):
        import numpy as np
        rng = np.random.default_rng(1)
        prices = _series((100 + rng.standard_normal(80).cumsum()).tolist())
        macd_line, signal_line, hist = compute_macd(prices)
        pd.testing.assert_series_equal(hist, macd_line - signal_line)

    def test_rising_trend_gives_positive_macd(self):
        # Strongly rising series: fast EMA > slow EMA → macd > 0
        prices = _series([float(i) * 2 for i in range(1, 60)])
        macd_line, _, _ = compute_macd(prices)
        assert float(macd_line.iloc[-1]) > 0

    def test_falling_trend_gives_negative_macd(self):
        prices = _series([float(60 - i) * 2 for i in range(60)])
        macd_line, _, _ = compute_macd(prices)
        assert float(macd_line.iloc[-1]) < 0

    def test_invalid_periods_raise(self):
        prices = _series([1.0] * 50)
        with pytest.raises(ValueError):
            compute_macd(prices, fast=0, slow=26, signal=9)
        with pytest.raises(ValueError):
            compute_macd(prices, fast=26, slow=12, signal=9)  # fast >= slow
