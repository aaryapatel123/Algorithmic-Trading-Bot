"""Unit tests for signal_generator sector cap and cat-stop warning logic."""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.signal_generator import apply_sector_cap, check_cat_stop_warnings

_SECTOR_MAP = {
    "AMAT": "Information Technology",
    "LRCX": "Information Technology",
    "AMD":  "Information Technology",
    "INTC": "Information Technology",
    "MU":   "Information Technology",
    "XOM":  "Energy",
    "CVX":  "Energy",
    "JPM":  "Financials",
}


# ---------------------------------------------------------------------------
# apply_sector_cap
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_sector_cap_clips_over_cap_sector():
    # 5 IT names at 17% each = 85% IT — exceeds 80% cap
    weights = {s: 0.17 for s in ["AMAT", "LRCX", "AMD", "INTC", "MU"]}
    result = apply_sector_cap(weights, _SECTOR_MAP, cap=0.80)
    it_total = sum(result[s] for s in result)
    assert abs(it_total - 0.80) < 1e-9, f"Expected IT total=0.80, got {it_total}"


@pytest.mark.unit
def test_sector_cap_preserves_relative_weights_within_sector():
    weights = {"AMAT": 0.30, "LRCX": 0.20, "AMD": 0.35}  # IT total = 0.85
    result = apply_sector_cap(weights, _SECTOR_MAP, cap=0.80)
    # Ratios between names must be preserved after scaling
    scale = 0.80 / 0.85
    assert abs(result["AMAT"] - 0.30 * scale) < 1e-9
    assert abs(result["LRCX"] - 0.20 * scale) < 1e-9
    assert abs(result["AMD"]  - 0.35 * scale) < 1e-9


@pytest.mark.unit
def test_sector_cap_no_op_when_under_cap():
    weights = {"AMAT": 0.20, "LRCX": 0.20, "XOM": 0.20}  # IT=40%, Energy=20%
    result = apply_sector_cap(weights, _SECTOR_MAP, cap=0.80)
    assert result == weights


@pytest.mark.unit
def test_sector_cap_at_exact_boundary_is_not_clipped():
    weights = {"AMAT": 0.40, "LRCX": 0.40}  # IT total = exactly 0.80
    result = apply_sector_cap(weights, _SECTOR_MAP, cap=0.80)
    assert abs(result["AMAT"] - 0.40) < 1e-9
    assert abs(result["LRCX"] - 0.40) < 1e-9


@pytest.mark.unit
def test_sector_cap_only_clips_over_cap_sector():
    # IT=85% (over cap), Energy=10% (under cap)
    weights = {
        "AMAT": 0.17, "LRCX": 0.17, "AMD": 0.17, "INTC": 0.17, "MU": 0.17,
        "XOM": 0.10,
    }
    result = apply_sector_cap(weights, _SECTOR_MAP, cap=0.80)
    it_total = sum(result[s] for s in ["AMAT", "LRCX", "AMD", "INTC", "MU"])
    assert abs(it_total - 0.80) < 1e-9
    assert abs(result["XOM"] - 0.10) < 1e-9, "Energy untouched"


@pytest.mark.unit
def test_sector_cap_passthrough_when_cap_is_one():
    weights = {"AMAT": 0.50, "LRCX": 0.50}
    result = apply_sector_cap(weights, _SECTOR_MAP, cap=1.0)
    assert result == weights


@pytest.mark.unit
def test_sector_cap_passthrough_on_empty_weights():
    assert apply_sector_cap({}, _SECTOR_MAP, cap=0.80) == {}


@pytest.mark.unit
def test_sector_cap_unknown_sector_treated_as_own_bucket():
    weights = {"UNKNOWN_TICKER": 0.90}
    result = apply_sector_cap(weights, _SECTOR_MAP, cap=0.80)
    assert abs(result["UNKNOWN_TICKER"] - 0.80) < 1e-9


# ---------------------------------------------------------------------------
# check_cat_stop_warnings
# ---------------------------------------------------------------------------

def _price_series(price: float) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=300, freq="D")
    return pd.Series([price] * 300, index=idx)


@pytest.mark.unit
def test_cat_stop_triggers_when_below_threshold():
    target_weights = {"AMAT": 0.20}
    entry_prices   = {"AMAT": 200.00}
    current_prices = {"AMAT": _price_series(155.00)}  # −22.5% < −20%
    warnings = check_cat_stop_warnings(target_weights, entry_prices, current_prices, threshold=0.20)
    assert len(warnings) == 1
    assert warnings[0]["symbol"] == "AMAT"
    assert warnings[0]["drawdown_pct"] < -20.0


@pytest.mark.unit
def test_cat_stop_no_trigger_within_threshold():
    target_weights = {"LRCX": 0.20}
    entry_prices   = {"LRCX": 200.00}
    current_prices = {"LRCX": _price_series(170.00)}  # −15% > −20%
    warnings = check_cat_stop_warnings(target_weights, entry_prices, current_prices, threshold=0.20)
    assert warnings == []


@pytest.mark.unit
def test_cat_stop_at_exact_threshold_does_not_trigger():
    target_weights = {"AMD": 0.17}
    entry_prices   = {"AMD": 100.00}
    current_prices = {"AMD": _price_series(80.00)}  # exactly −20%
    warnings = check_cat_stop_warnings(target_weights, entry_prices, current_prices, threshold=0.20)
    assert warnings == []


@pytest.mark.unit
def test_cat_stop_ignores_zero_weight_positions():
    # Name no longer held (weight=0) — should not warn even if price collapsed
    target_weights = {"INTC": 0.0}
    entry_prices   = {"INTC": 200.00}
    current_prices = {"INTC": _price_series(50.00)}  # −75%
    warnings = check_cat_stop_warnings(target_weights, entry_prices, current_prices)
    assert warnings == []


@pytest.mark.unit
def test_cat_stop_ignores_missing_current_price():
    target_weights = {"MU": 0.15}
    entry_prices   = {"MU": 100.00}
    current_prices = {}  # no price data
    warnings = check_cat_stop_warnings(target_weights, entry_prices, current_prices)
    assert warnings == []


@pytest.mark.unit
def test_cat_stop_ignores_missing_entry_price():
    target_weights = {"MU": 0.15}
    entry_prices   = {}  # no stored entry price
    current_prices = {"MU": _price_series(50.00)}
    warnings = check_cat_stop_warnings(target_weights, entry_prices, current_prices)
    assert warnings == []


@pytest.mark.unit
def test_cat_stop_multiple_names_only_warns_on_breached():
    target_weights = {"AMAT": 0.20, "LRCX": 0.20, "AMD": 0.17}
    entry_prices   = {"AMAT": 200.00, "LRCX": 200.00, "AMD": 100.00}
    current_prices = {
        "AMAT": _price_series(155.00),  # −22.5% → warns
        "LRCX": _price_series(175.00),  # −12.5% → ok
        "AMD":  _price_series(70.00),   # −30%   → warns
    }
    warnings = check_cat_stop_warnings(target_weights, entry_prices, current_prices, threshold=0.20)
    warned_syms = {w["symbol"] for w in warnings}
    assert warned_syms == {"AMAT", "AMD"}


@pytest.mark.unit
def test_cat_stop_respects_custom_threshold():
    target_weights = {"XOM": 0.10}
    entry_prices   = {"XOM": 100.00}
    current_prices = {"XOM": _price_series(78.00)}  # −22%
    # With 25% threshold, −22% should NOT trigger
    assert check_cat_stop_warnings(target_weights, entry_prices, current_prices, threshold=0.25) == []
    # With 20% threshold, −22% SHOULD trigger
    assert len(check_cat_stop_warnings(target_weights, entry_prices, current_prices, threshold=0.20)) == 1
