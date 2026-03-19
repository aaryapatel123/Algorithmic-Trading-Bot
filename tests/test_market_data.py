from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.market_data import AlpacaDataProvider


def _make_mock_bar(o=100.0, h=101.0, l=99.0, c=100.5, v=1000, t="2024-01-01T00:00:00Z"):
    bar = MagicMock()
    bar.o = o
    bar.h = h
    bar.l = l
    bar.c = c
    bar.v = v
    bar.t = t
    return bar


def _make_rest_client(bars):
    rest = MagicMock()
    rest.get_barset.return_value = {"AAPL": bars}
    return rest


@pytest.fixture
def provider_with_bars():
    bars = [_make_mock_bar(t=f"2024-01-{i+1:02d}T00:00:00Z") for i in range(60)]
    rest = _make_rest_client(bars)
    return AlpacaDataProvider(rest), rest


def test_get_historical_bars_returns_dataframe(provider_with_bars):
    provider, _ = provider_with_bars
    df = provider.get_historical_bars("AAPL", "1D", 60)
    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_get_historical_bars_sorted_ascending(provider_with_bars):
    provider, _ = provider_with_bars
    df = provider.get_historical_bars("AAPL", "1D", 60)
    assert df.index.is_monotonic_increasing


def test_get_historical_bars_calls_correct_timeframe():
    rest = _make_rest_client([_make_mock_bar()])
    provider = AlpacaDataProvider(rest)
    provider.get_historical_bars("AAPL", "1D", 10)
    rest.get_barset.assert_called_once_with("AAPL", "day", limit=10)


def test_empty_alpaca_response_falls_back_to_yfinance():
    rest = _make_rest_client([])
    provider = AlpacaDataProvider(rest)

    fake_df = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1000]},
        index=pd.date_range("2024-01-01", periods=1, tz="UTC"),
    )

    with patch("src.data.market_data.yf") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_df
        mock_yf.Ticker.return_value = mock_ticker

        df = provider.get_historical_bars("AAPL", "1D", 1)
        assert not df.empty


def test_get_latest_price_returns_float(provider_with_bars):
    provider, _ = provider_with_bars
    price = provider.get_latest_price("AAPL")
    assert isinstance(price, float)
    assert price > 0


def test_alpaca_failure_retries_then_falls_back():
    rest = MagicMock()
    rest.get_barset.side_effect = Exception("API timeout")
    provider = AlpacaDataProvider(rest)

    fake_df = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1000]},
        index=pd.date_range("2024-01-01", periods=1, tz="UTC"),
    )

    with patch("src.data.market_data.time.sleep"), patch("src.data.market_data.yf") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_df
        mock_yf.Ticker.return_value = mock_ticker

        df = provider.get_historical_bars("AAPL", "1D", 1)
        assert not df.empty
        assert rest.get_barset.call_count == 3  # retried 3 times
