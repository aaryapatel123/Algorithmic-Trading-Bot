"""Adaptive-signal primitives — Pillar 1 (dynamic signal tuning).

Pure, identifier-agnostic helpers for two signal extensions layered on top of
the proven keep8 + smooth 0.5 + cap25 book *without* touching selection,
weighting, capping or smoothing:

  • momentum acceleration — a second-derivative tilt that de-weights stocks
    whose 12-month return is decelerating (rolling over) before the fixed
    252-day window mechanically drops them. Targets the 2021 rotation bleed.

  • rolling Information Coefficient (IC) horizon weighting — measures which
    lookback (1/3/6/12-month) has actually been predictive lately and weights
    a multi-horizon blend by that, so a decaying horizon is de-emphasised
    automatically rather than by a fixed guess.

Everything here is a pure function of its arguments (no price access, no clock,
no I/O) so the look-ahead-sensitive maths is unit-testable in isolation. The
stateful orchestration (what is "prior", when IC updates) lives in the strategy,
which is the only place that knows the rebalance clock.
"""
from __future__ import annotations

import math
from typing import Hashable, Sequence


def cross_sectional_zscore(values: dict[Hashable, float]) -> dict[Hashable, float]:
    """Standardise a cross-section to mean 0, std 1.

    Uses the population std (ddof=0) — we are describing the dispersion of the
    sample in front of us, not inferring a wider population. Returns all-zeros
    when there are fewer than two names or the cross-section is degenerate
    (zero dispersion), so a flat signal contributes nothing rather than blowing
    up. Keys are preserved.
    """
    if not values:
        return {}
    xs = list(values.values())
    n = len(xs)
    if n < 2:
        return {k: 0.0 for k in values}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(var)
    if std <= 0:
        return {k: 0.0 for k in values}
    return {k: (v - mean) / std for k, v in values.items()}


def _average_ranks(xs: Sequence[float]) -> list[float]:
    """Return 1-based ranks of ``xs`` ascending, averaging ties."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation of two equal-length series.

    Computed as the Pearson correlation of average ranks (so ties are handled
    correctly). Returns 0.0 for degenerate input (length < 2, mismatched
    lengths, or zero rank-variance on either side) — i.e. "no measurable
    relationship", which is the safe neutral value for the IC tracker.
    """
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def softmax(scores: dict[Hashable, float], tau: float) -> dict[Hashable, float]:
    """Temperature-scaled softmax over ``scores`` (numerically stable).

    ``tau`` is the temperature: smaller → sharper (concentrates weight on the
    largest score), larger → flatter. An all-equal input (e.g. all-zero clamped
    ICs) returns uniform weights, which is exactly the desired neutral default
    before any horizon has proven itself. ``tau <= 0`` collapses to argmax-ish
    behaviour via a tiny floor.
    """
    if not scores:
        return {}
    t = max(tau, 1e-9)
    keys = list(scores)
    mx = max(scores[k] for k in keys)
    exps = {k: math.exp((scores[k] - mx) / t) for k in keys}
    total = sum(exps.values())
    if total <= 0:
        u = 1.0 / len(keys)
        return {k: u for k in keys}
    return {k: e / total for k, e in exps.items()}


class ICTracker:
    """EWMA of each horizon's rolling Information Coefficient.

    The tracker holds one smoothed IC per lookback horizon. ``update`` folds in
    the latest realised IC (one observation per rebalance) with memory ``beta``
    (higher = more reactive). ``weights`` clamps negative ICs to zero — a
    long-only book cannot cleanly exploit an anti-predictive horizon, so it is
    silenced rather than inverted — and maps the survivors through a softmax.

    Determinism note: with no updates, or with all ICs ≤ 0, ``weights`` returns
    uniform weights, so the blend degrades gracefully to an equal-weight
    multi-horizon signal (never to a single arbitrary horizon).
    """

    def __init__(self, horizons: Sequence[int], beta: float = 0.2) -> None:
        if not horizons:
            raise ValueError("ICTracker needs at least one horizon")
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        self.horizons: tuple[int, ...] = tuple(horizons)
        self.beta = beta
        self.ic_ewma: dict[int, float] = {h: 0.0 for h in self.horizons}
        self.n_updates: int = 0

    def update(self, ic_by_horizon: dict[int, float]) -> None:
        """Fold one rebalance's realised ICs into the EWMA (missing → skip)."""
        updated = False
        for h in self.horizons:
            if h not in ic_by_horizon:
                continue
            ic = ic_by_horizon[h]
            self.ic_ewma[h] = (1.0 - self.beta) * self.ic_ewma[h] + self.beta * ic
            updated = True
        if updated:
            self.n_updates += 1

    def weights(self, tau: float = 0.10) -> dict[int, float]:
        """Softmax weights over positive-clamped smoothed ICs (sum to 1)."""
        clamped = {h: max(self.ic_ewma[h], 0.0) for h in self.horizons}
        return softmax(clamped, tau)
