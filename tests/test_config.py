from __future__ import annotations

import os

import pytest


def test_load_config_reads_env_vars(sample_config):
    assert sample_config.api_key_id == "test_key"
    assert sample_config.api_secret_key == "test_secret"
    assert sample_config.symbols == ["AAPL", "MSFT"]
    assert sample_config.short_ma_period == 5
    assert sample_config.long_ma_period == 10


def test_config_defaults(sample_config):
    assert sample_config.bar_timeframe == "1D"
    assert sample_config.log_level == "WARNING"
    assert sample_config.max_position_pct == 0.10
    assert sample_config.max_total_exposure_pct == 0.50


def test_config_is_immutable(sample_config):
    with pytest.raises((AttributeError, TypeError)):
        sample_config.symbols = ["GOOG"]


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "")
    from src import config as cfg_module
    import importlib
    importlib.reload(cfg_module)
    with pytest.raises(ValueError, match="APCA_API_KEY_ID"):
        cfg_module.load_config()


def test_short_gte_long_raises(monkeypatch):
    monkeypatch.setenv("SHORT_MA_PERIOD", "50")
    monkeypatch.setenv("LONG_MA_PERIOD", "20")
    from src import config as cfg_module
    import importlib
    importlib.reload(cfg_module)
    with pytest.raises(ValueError, match="SHORT_MA_PERIOD"):
        cfg_module.load_config()


def test_empty_symbols_raises(monkeypatch):
    monkeypatch.setenv("SYMBOLS", "")
    from src import config as cfg_module
    import importlib
    importlib.reload(cfg_module)
    with pytest.raises(ValueError, match="SYMBOLS"):
        cfg_module.load_config()
