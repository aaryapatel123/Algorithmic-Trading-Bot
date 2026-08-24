"""Holding hysteresis + EWMA weight smoothing — friction-reduction primitives.

Both functions are pure and identifier-agnostic (they operate on any hashable
that stands for a holding — a backtrader feed, a ticker string, …), so they unit
test without a backtest and drop cleanly into the selection/weighting steps of
``bt_inv_vol.py``.

Why this sits in the "proven lane" (SESSION_NOTES_4 / IMPLEMENTATION.md):
    Every lever that *broadened* or *reweighted* the concentrated momentum book
    (top_n sweep, ERC, tranches) diluted the alpha. These two do the opposite —
    they keep the book at exactly ``top_n`` and only cut *turnover*:

      • hysteresis reduces churn in *which* names are held (a rank buffer: drop a
        name only once it leaves the top ``keep_n``, not the instant it leaves
        top_n), so a name oscillating around the rank-N boundary isn't traded
        every month;
      • smoothing reduces churn in *how much* of each name is held (blend new
        inverse-vol targets toward the prior weights), damping vol-estimate noise.

    Session 1 already showed weekly rebalancing *hurt* the book via friction and
    premature exits — direct evidence that cutting turnover should help net (and
    after-tax) return without touching market exposure or de-selecting winners.
"""
from __future__ import annotations

from typing import Hashable, Iterable, Sequence


def apply_hysteresis(
    ranked: Sequence[Hashable],
    held: Iterable[Hashable],
    top_n: int,
    keep_n: int,
) -> list[Hashable]:
    """Select ``top_n`` names from ``ranked`` (best→worst) with a rank buffer.

    Rule ("exit buffer", a.k.a. drop-only-when-it-leaves-the-top-``keep_n``):

      1. Retain incumbents — names in ``held`` — that are still inside the top
         ``keep_n`` of the ranking, best-ranked first, up to ``top_n`` slots.
      2. Fill any slots left over (from incumbents that fell out of the buffer)
         with the best not-yet-selected names.

    A held name therefore stays in the book while its rank ∈ [1, keep_n] even if
    it has slipped out of the strict top_n, so a name hovering around the rank-N
    cutoff is not bought/sold every rebalance. ``keep_n <= top_n`` disables the
    buffer and reproduces the naive ``ranked[:top_n]`` exactly.

    The result is returned in best→worst rank order (so downstream conviction
    weighting stays valid) and has length ``min(top_n, len(ranked))``.
    """
    ranked = list(ranked)
    if keep_n <= top_n or top_n <= 0:
        return ranked[:top_n]

    held_set = set(held)
    buffer_window = ranked[:keep_n]
    selected = set(d for d in buffer_window if d in held_set)

    # Cap retained incumbents to top_n best-ranked (rank order == ranked order).
    if len(selected) > top_n:
        selected = set(d for d in ranked if d in selected)  # rank-ordered dedupe
        selected = set(list(_in_rank_order(ranked, selected))[:top_n])

    # Fill remaining slots with the best names not already chosen.
    for d in ranked:
        if len(selected) >= top_n:
            break
        selected.add(d)

    return _in_rank_order(ranked, selected)[:top_n]


def _in_rank_order(ranked: Sequence[Hashable], names: set) -> list[Hashable]:
    """Filter ``names`` back into ``ranked`` order (best→worst)."""
    return [d for d in ranked if d in names]


def smooth_weights(
    target: dict[Hashable, float],
    prev: dict[Hashable, float],
    alpha: float,
) -> dict[Hashable, float]:
    """EWMA-blend freshly computed ``target`` weights toward the prior weights.

    ``w = (1-alpha)·prev + alpha·target`` per name, then renormalised so the
    total invested budget (``sum(target)``) is preserved — smoothing changes the
    *path* toward the target, not total market exposure.

    Only names present in ``target`` (the current book) receive weight; a name
    dropped from the book gets nothing, so smoothing never keeps a sold name
    alive. ``alpha >= 1`` is a no-op (returns the target unchanged). With an
    empty ``prev`` (first rebalance) the uniform ``alpha`` scaling cancels in the
    renormalisation, so the target is applied unchanged.
    """
    if alpha >= 1.0 or not target:
        return dict(target)

    budget = sum(target.values())
    blended = {
        d: (1.0 - alpha) * prev.get(d, 0.0) + alpha * w
        for d, w in target.items()
    }
    total = sum(blended.values())
    if total <= 0:
        return dict(target)

    scale = budget / total
    return {d: w * scale for d, w in blended.items()}
