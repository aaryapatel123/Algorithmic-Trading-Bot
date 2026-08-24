"""Unit tests for backtest._signal (Pillar 1 adaptive-signal primitives)."""
from __future__ import annotations

import math

import pytest

from backtest._signal import (
    ICTracker,
    cross_sectional_zscore,
    softmax,
    spearman_corr,
)


# ---------------------------------------------------------------------------
# cross_sectional_zscore
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_zscore_basic_mean_zero_std_one():
    z = cross_sectional_zscore({"a": 1.0, "b": 2.0, "c": 3.0})
    vals = list(z.values())
    assert math.isclose(sum(vals), 0.0, abs_tol=1e-12)
    # population std of standardised values is 1
    mean = sum(vals) / len(vals)
    pop_std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    assert math.isclose(pop_std, 1.0, rel_tol=1e-9)


@pytest.mark.unit
def test_zscore_preserves_order():
    z = cross_sectional_zscore({"lo": -5.0, "mid": 0.0, "hi": 10.0})
    assert z["hi"] > z["mid"] > z["lo"]


@pytest.mark.unit
def test_zscore_degenerate_returns_zeros():
    assert cross_sectional_zscore({}) == {}
    assert cross_sectional_zscore({"a": 7.0}) == {"a": 0.0}
    # zero dispersion → all zeros, not a divide-by-zero
    assert cross_sectional_zscore({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}


# ---------------------------------------------------------------------------
# spearman_corr
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_spearman_perfect_monotonic():
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    assert math.isclose(spearman_corr(x, y), 1.0, rel_tol=1e-9)


@pytest.mark.unit
def test_spearman_perfect_inverse():
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    assert math.isclose(spearman_corr(x, y), -1.0, rel_tol=1e-9)


@pytest.mark.unit
def test_spearman_nonlinear_monotonic_is_one():
    # Spearman (unlike Pearson) is 1 for any monotonic map
    x = [1, 2, 3, 4]
    y = [1, 4, 9, 16]
    assert math.isclose(spearman_corr(x, y), 1.0, rel_tol=1e-9)


@pytest.mark.unit
def test_spearman_handles_ties():
    x = [1, 1, 2, 3]
    y = [2, 2, 5, 9]
    # both have a tie in the same place → still perfectly monotonic
    assert math.isclose(spearman_corr(x, y), 1.0, rel_tol=1e-9)


@pytest.mark.unit
def test_spearman_degenerate_returns_zero():
    assert spearman_corr([1.0], [1.0]) == 0.0
    assert spearman_corr([1, 2, 3], [1, 2]) == 0.0  # mismatched length
    assert spearman_corr([5, 5, 5], [1, 2, 3]) == 0.0  # zero variance side


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_softmax_sums_to_one():
    w = softmax({"a": 1.0, "b": 2.0, "c": 3.0}, tau=1.0)
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-12)
    assert w["c"] > w["b"] > w["a"]


@pytest.mark.unit
def test_softmax_uniform_on_equal_input():
    w = softmax({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}, tau=0.1)
    for v in w.values():
        assert math.isclose(v, 0.25, rel_tol=1e-9)


@pytest.mark.unit
def test_softmax_lower_tau_sharper():
    hot = softmax({"a": 0.0, "b": 1.0}, tau=0.1)
    cool = softmax({"a": 0.0, "b": 1.0}, tau=2.0)
    # lower temperature concentrates more weight on the larger score
    assert hot["b"] > cool["b"]


@pytest.mark.unit
def test_softmax_numerically_stable_large_scores():
    w = softmax({"a": 1000.0, "b": 1001.0}, tau=1.0)
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-12)
    assert w["b"] > w["a"]


# ---------------------------------------------------------------------------
# ICTracker
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ic_tracker_initial_weights_uniform():
    t = ICTracker((21, 63, 126, 252), beta=0.2)
    w = t.weights()
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-12)
    for v in w.values():
        assert math.isclose(v, 0.25, rel_tol=1e-9)


@pytest.mark.unit
def test_ic_tracker_ewma_moves_toward_observation():
    t = ICTracker((21, 252), beta=0.5)
    t.update({21: 0.4, 252: 0.0})
    assert math.isclose(t.ic_ewma[21], 0.2, rel_tol=1e-9)  # 0.5*0.4
    t.update({21: 0.4, 252: 0.0})
    assert math.isclose(t.ic_ewma[21], 0.3, rel_tol=1e-9)  # 0.5*0.2 + 0.5*0.4


@pytest.mark.unit
def test_ic_tracker_positive_horizon_gets_more_weight():
    t = ICTracker((21, 252), beta=0.5)
    for _ in range(5):
        t.update({21: -0.3, 252: 0.5})  # 21d anti-predictive, 252d predictive
    w = t.weights(tau=0.1)
    assert w[252] > w[21]


@pytest.mark.unit
def test_ic_tracker_clamps_negative_to_uniform():
    t = ICTracker((21, 252), beta=0.5)
    for _ in range(5):
        t.update({21: -0.3, 252: -0.5})  # both anti-predictive
    w = t.weights(tau=0.1)
    # both clamped to 0 → uniform, never inverts into a short-style bet
    assert math.isclose(w[21], 0.5, rel_tol=1e-9)
    assert math.isclose(w[252], 0.5, rel_tol=1e-9)


@pytest.mark.unit
def test_ic_tracker_skips_missing_horizon():
    t = ICTracker((21, 252), beta=0.5)
    t.update({21: 0.4})  # 252 missing this rebalance
    assert math.isclose(t.ic_ewma[21], 0.2, rel_tol=1e-9)
    assert t.ic_ewma[252] == 0.0  # untouched


@pytest.mark.unit
def test_ic_tracker_validates_args():
    with pytest.raises(ValueError):
        ICTracker((), beta=0.2)
    with pytest.raises(ValueError):
        ICTracker((21,), beta=0.0)
    with pytest.raises(ValueError):
        ICTracker((21,), beta=1.5)
