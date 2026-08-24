"""Unit tests for holding hysteresis + EWMA weight smoothing.

Both functions are pure and identifier-agnostic (tested with plain strings as
stand-ins for backtrader data feeds), so the churn-reduction logic can be
verified without spinning up a backtest.
"""
from __future__ import annotations

import pytest

from backtest._hysteresis import apply_hysteresis, smooth_weights


# ---------------------------------------------------------------------------
# apply_hysteresis — selection with a rank buffer
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_hysteresis_when_keep_n_not_greater_than_top_n():
    """keep_n <= top_n must reproduce the naive top_n selection exactly."""
    ranked = ["A", "B", "C", "D", "E", "F", "G"]
    held = {"F", "G"}
    assert apply_hysteresis(ranked, held, top_n=5, keep_n=5) == ["A", "B", "C", "D", "E"]
    assert apply_hysteresis(ranked, held, top_n=5, keep_n=0) == ["A", "B", "C", "D", "E"]


@pytest.mark.unit
def test_first_period_with_no_holdings_is_naive_topn():
    """An empty book (first rebalance) selects the plain top_n."""
    ranked = ["A", "B", "C", "D", "E", "F", "G", "H"]
    assert apply_hysteresis(ranked, set(), top_n=5, keep_n=7) == ["A", "B", "C", "D", "E"]


@pytest.mark.unit
def test_incumbents_in_buffer_are_retained_over_fresh_names():
    """Incumbents still inside the keep_n buffer block marginally-better fresh
    entrants — the core churn-reduction behaviour (zero trades here)."""
    ranked = ["A", "B", "C", "D", "E", "F", "G"]
    held = {"A", "B", "C", "F", "G"}  # F,G slipped to rank 6,7 but still in top-7
    # D,E (fresh, rank 4,5) do NOT displace F,G — book unchanged.
    assert apply_hysteresis(ranked, held, top_n=5, keep_n=7) == ["A", "B", "C", "F", "G"]


@pytest.mark.unit
def test_name_dropped_only_when_it_exits_the_buffer():
    """A held name ranked beyond keep_n is sold; the freed slot goes to the best
    available fresh name."""
    ranked = ["A", "B", "C", "D", "E", "F", "G", "H"]
    held = {"D", "E", "F", "G", "H"}  # H is rank 8 → outside keep_n=7 → must exit
    # H exits; one slot opens; best non-selected (A) enters. D,E,F,G retained.
    assert apply_hysteresis(ranked, held, top_n=5, keep_n=7) == ["A", "D", "E", "F", "G"]


@pytest.mark.unit
def test_more_incumbents_than_slots_keeps_best_ranked():
    """When more incumbents sit inside the buffer than there are slots, keep the
    best-ranked top_n of them and drop the rest."""
    ranked = ["A", "B", "C", "D", "E", "F", "G"]
    held = {"A", "B", "C", "D", "E", "F", "G"}  # all 7 held
    assert apply_hysteresis(ranked, held, top_n=5, keep_n=7) == ["A", "B", "C", "D", "E"]


@pytest.mark.unit
def test_result_is_in_rank_order_and_correct_size():
    """Output preserves best→worst rank order (so momentum_rank_weights stays
    valid) and is exactly top_n long."""
    ranked = ["A", "B", "C", "D", "E", "F", "G", "H"]
    held = {"C", "F"}
    out = apply_hysteresis(ranked, held, top_n=4, keep_n=7)
    assert len(out) == 4
    assert out == sorted(out, key=ranked.index)


@pytest.mark.unit
def test_fewer_candidates_than_top_n_returns_all():
    ranked = ["A", "B"]
    assert apply_hysteresis(ranked, {"A"}, top_n=5, keep_n=8) == ["A", "B"]


# ---------------------------------------------------------------------------
# smooth_weights — EWMA blend toward previous weights
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_alpha_one_is_no_smoothing():
    target = {"A": 0.5, "B": 0.3, "C": 0.2}
    prev = {"A": 0.1, "B": 0.1, "C": 0.8}
    assert smooth_weights(target, prev, alpha=1.0) == target


@pytest.mark.unit
def test_empty_prev_returns_target_after_renormalisation():
    """First period (no prior weights) must apply the target unchanged: the
    renormalisation cancels the uniform alpha scaling."""
    target = {"A": 0.6, "B": 0.4}
    out = smooth_weights(target, {}, alpha=0.5)
    assert out["A"] == pytest.approx(0.6)
    assert out["B"] == pytest.approx(0.4)


@pytest.mark.unit
def test_smoothing_preserves_total_budget():
    """Smoothing changes the path, not total exposure: sum is preserved."""
    target = {"A": 0.4, "B": 0.3, "C": 0.1}  # budget 0.8 (e.g. after cash buffer)
    prev = {"A": 0.1, "B": 0.5, "C": 0.2}
    out = smooth_weights(target, prev, alpha=0.5)
    assert sum(out.values()) == pytest.approx(0.8)


@pytest.mark.unit
def test_smoothing_pulls_toward_previous_weight():
    """A name whose target jumped up is dampened toward its previous weight."""
    target = {"A": 0.8, "B": 0.2}
    prev = {"A": 0.2, "B": 0.8}
    out = smooth_weights(target, prev, alpha=0.5)
    # A's applied weight should sit between prev (0.2) and target (0.8), below target.
    assert prev["A"] < out["A"] < target["A"]


@pytest.mark.unit
def test_dropped_name_is_not_carried_over():
    """A name no longer in the book (sold) gets no weight, even if it had a large
    previous weight — smoothing must never keep a dropped name alive."""
    target = {"A": 0.5, "B": 0.5}
    prev = {"A": 0.2, "B": 0.2, "C": 0.6}  # C was held, now dropped
    out = smooth_weights(target, prev, alpha=0.5)
    assert "C" not in out
