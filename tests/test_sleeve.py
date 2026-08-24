"""Unit tests for Pillar 4 — diversifying sleeve (_sleeve_weights logic).

Tests the budget allocation and weighting logic in isolation using pure
functions that mirror the strategy internals, without running a full backtest.
"""
from __future__ import annotations

import math
import unittest


# ---------------------------------------------------------------------------
# Pure reimplementations of _sleeve_weights internals for unit testing
# ---------------------------------------------------------------------------

def _sleeve_weights(
    sleeve_pct: float,
    asset_vols: dict[str, float] | None,  # name → annualised vol; None = no inv_vol
    sleeve_weighting: str,
    budget: float | None = None,
) -> dict[str, float]:
    """Reimplementation of _sleeve_weights for testing.

    asset_vols keys determine the sleeve assets; values are their vols.
    Pass asset_vols=None or empty to test the off/no-history path.
    """
    feeds = list(asset_vols.keys()) if asset_vols else []
    if sleeve_pct <= 0.0 or not feeds:
        return {}

    b = budget if budget is not None else sleeve_pct

    if sleeve_weighting == "inv_vol" and len(feeds) > 1 and asset_vols:
        valid = {k: v for k, v in asset_vols.items() if v > 0}
        if len(valid) == len(feeds):
            total_inv = sum(1.0 / v for v in valid.values())
            return {k: (1.0 / v) / total_inv * b for k, v in valid.items()}

    per_asset = b / len(feeds)
    return {k: per_asset for k in feeds}


def _equity_investable(cash_buffer: float, sleeve_pct: float, vol_scalar: float = 1.0) -> float:
    return max(0.0, 1.0 - cash_buffer - sleeve_pct) * vol_scalar


class TestSleeveOff(unittest.TestCase):
    def test_zero_sleeve_pct_returns_empty(self):
        result = _sleeve_weights(0.0, {"AGG": 0.05, "GLD": 0.12}, "equal")
        self.assertEqual(result, {})

    def test_no_assets_returns_empty(self):
        result = _sleeve_weights(0.10, {}, "equal")
        self.assertEqual(result, {})

    def test_negative_sleeve_pct_returns_empty(self):
        result = _sleeve_weights(-0.05, {"GLD": 0.12}, "equal")
        self.assertEqual(result, {})


class TestEqualWeighting(unittest.TestCase):
    def test_single_asset_gets_full_budget(self):
        result = _sleeve_weights(0.10, {"GLD": 0.12}, "equal")
        self.assertAlmostEqual(result["GLD"], 0.10)

    def test_three_assets_equal_split(self):
        result = _sleeve_weights(0.15, {"AGG": 0.05, "GLD": 0.12, "TLT": 0.10}, "equal")
        self.assertEqual(len(result), 3)
        for w in result.values():
            self.assertAlmostEqual(w, 0.05)

    def test_weights_sum_to_sleeve_pct(self):
        result = _sleeve_weights(0.20, {"AGG": 0.05, "GLD": 0.12, "TLT": 0.10}, "equal")
        self.assertAlmostEqual(sum(result.values()), 0.20, places=10)

    def test_two_assets_equal_split(self):
        result = _sleeve_weights(0.10, {"GLD": 0.12, "TLT": 0.10}, "equal")
        self.assertAlmostEqual(result["GLD"], 0.05)
        self.assertAlmostEqual(result["TLT"], 0.05)


class TestInvVolWeighting(unittest.TestCase):
    def test_lower_vol_gets_higher_weight(self):
        # AGG vol=0.05 (low), GLD vol=0.15 (high) → AGG gets more
        result = _sleeve_weights(0.10, {"AGG": 0.05, "GLD": 0.15}, "inv_vol")
        self.assertGreater(result["AGG"], result["GLD"])

    def test_weights_sum_to_sleeve_pct(self):
        result = _sleeve_weights(0.15, {"AGG": 0.05, "GLD": 0.12, "TLT": 0.10}, "inv_vol")
        self.assertAlmostEqual(sum(result.values()), 0.15, places=10)

    def test_equal_vol_gives_equal_weights(self):
        result = _sleeve_weights(0.10, {"AGG": 0.10, "GLD": 0.10}, "inv_vol")
        self.assertAlmostEqual(result["AGG"], result["GLD"])

    def test_single_asset_falls_back_to_full_budget(self):
        # Single asset: inv_vol branch skipped (len <= 1), falls to equal
        result = _sleeve_weights(0.10, {"GLD": 0.12}, "inv_vol")
        self.assertAlmostEqual(result["GLD"], 0.10)

    def test_zero_vol_asset_falls_back_to_equal(self):
        # Zero vol means inv_vol solve fails → falls back to equal
        result = _sleeve_weights(0.10, {"AGG": 0.0, "GLD": 0.12}, "inv_vol")
        self.assertAlmostEqual(sum(result.values()), 0.10, places=10)
        # Equal fallback: 0.05 each
        self.assertAlmostEqual(result["AGG"], 0.05)
        self.assertAlmostEqual(result["GLD"], 0.05)


class TestEquityBudgetReduction(unittest.TestCase):
    def test_no_sleeve_full_budget(self):
        investable = _equity_investable(cash_buffer=0.05, sleeve_pct=0.0)
        self.assertAlmostEqual(investable, 0.95)

    def test_10pct_sleeve_reduces_equity(self):
        investable = _equity_investable(cash_buffer=0.05, sleeve_pct=0.10)
        self.assertAlmostEqual(investable, 0.85)

    def test_20pct_sleeve_reduces_equity(self):
        investable = _equity_investable(cash_buffer=0.05, sleeve_pct=0.20)
        self.assertAlmostEqual(investable, 0.75)

    def test_sleeve_plus_cash_over_100_clamps_to_zero(self):
        # sleeve_pct=0.96 + cash_buffer=0.05 > 1.0 → max(0, ...) = 0
        investable = _equity_investable(cash_buffer=0.05, sleeve_pct=0.96)
        self.assertAlmostEqual(investable, 0.0)

    def test_vol_scalar_applied_after_reduction(self):
        investable = _equity_investable(cash_buffer=0.05, sleeve_pct=0.10, vol_scalar=0.8)
        self.assertAlmostEqual(investable, 0.85 * 0.8)

    def test_total_allocation_sums_to_one(self):
        sleeve_pct = 0.15
        cash = 0.05
        equity = _equity_investable(cash, sleeve_pct)
        sleeve = sleeve_pct
        # equity + sleeve + cash ≈ 1.0
        self.assertAlmostEqual(equity + sleeve + cash, 1.0, places=10)


class TestWeightAllocation(unittest.TestCase):
    def test_weights_all_nonnegative(self):
        result = _sleeve_weights(0.15, {"AGG": 0.05, "GLD": 0.12, "TLT": 0.09}, "inv_vol")
        for w in result.values():
            self.assertGreaterEqual(w, 0.0)

    def test_all_weights_finite(self):
        result = _sleeve_weights(0.10, {"AGG": 0.05, "GLD": 0.12}, "inv_vol")
        for w in result.values():
            self.assertTrue(math.isfinite(w))


if __name__ == "__main__":
    unittest.main()
