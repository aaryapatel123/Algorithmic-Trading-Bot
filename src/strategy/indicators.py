from __future__ import annotations

"""Pure-pandas implementations of RSI, Bollinger Bands, and MACD."""

import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI using exponential smoothing (com = period - 1).
    Returns a Series of the same length as *close*; first `period` values are NaN.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 and avg_gain >= 0, RSI is defined as 100
    rsi = rsi.where(avg_loss != 0, other=100.0)
    return rsi


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (upper, mid, lower) Bollinger Bands.
    Uses population std (ddof=0) consistent with most charting platforms.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (macd_line, signal_line, histogram).
    All three series share the same index as *close*.
    """
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("fast, slow, and signal periods must all be positive")
    if fast >= slow:
        raise ValueError("fast period must be less than slow period")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
