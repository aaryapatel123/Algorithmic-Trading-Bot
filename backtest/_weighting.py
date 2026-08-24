"""Covariance-aware portfolio weighting: Ledoit-Wolf shrinkage + ERC.

Pure-numpy, no extra dependencies. Drop-in replacement for the inverse-volatility
weighting step in the momentum strategy.

Why ERC (equal risk contribution):
    Inverse-vol weighting treats holdings as independent — it ignores that the
    top momentum names are often highly correlated (e.g. all mega-cap tech in
    2020-21). ERC generalises inverse-vol: under a *diagonal* covariance (zero
    correlation) the ERC solution collapses exactly to inverse-vol. Feeding it
    the full covariance lowers portfolio vol via the correlation structure that
    inverse-vol ignores — without de-selecting any volatile winner.

Why shrinkage:
    With ~5 names and a short lookback the sample covariance is badly
    conditioned; raw min-variance/ERC on it becomes a bet on estimation error.
    Ledoit-Wolf (2004) shrinks the sample covariance toward a constant-correlation
    target with an analytically-optimal intensity, which stabilises the estimate.
"""
from __future__ import annotations

import numpy as np


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of a T×N returns matrix (rows = observations)."""
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        n = X.shape[1] if X.ndim == 2 else 1
        return np.eye(n)
    return np.cov(X, rowvar=False, ddof=1)


def ledoit_wolf_constant_corr(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage toward a constant-correlation target.

    ``returns`` is a T×N matrix (rows = observations, columns = assets).
    Returns the N×N shrunk covariance estimate ``delta*F + (1-delta)*S`` where
    ``S`` is the sample covariance, ``F`` the constant-correlation target, and
    ``delta`` the analytically-optimal shrinkage intensity clamped to [0, 1].
    """
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2:
        raise ValueError("returns must be a 2-D T×N matrix")
    t, n = X.shape
    if t < 2 or n == 0:
        var = np.var(X, axis=0, ddof=1) if t > 1 else np.ones(max(n, 1))
        return np.diag(np.atleast_1d(var))
    if n == 1:
        return np.array([[np.var(X[:, 0], ddof=1)]])

    Xc = X - X.mean(axis=0)
    s = (Xc.T @ Xc) / t                       # sample cov (1/T, per LW)
    var = np.diag(s).copy()
    std = np.sqrt(np.where(var > 0, var, 0.0))
    std_safe = np.where(std > 0, std, 1.0)

    # Constant-correlation target F.
    corr = s / np.outer(std_safe, std_safe)
    rbar = corr[np.triu_indices(n, k=1)].mean()
    f = rbar * np.outer(std_safe, std_safe)
    np.fill_diagonal(f, var)

    # pi: sum of asymptotic variances of the sample covariance entries.
    Xc2 = Xc ** 2
    pi_mat = (Xc2.T @ Xc2) / t - s ** 2
    pi_hat = pi_mat.sum()

    # rho: asymptotic covariance between target and sample estimators.
    rho_diag = np.trace(pi_mat)
    # theta_ii,ij = (1/T) sum_t A_ti^3 A_tj - s_ii * s_ij
    m3 = (Xc ** 3).T @ Xc / t
    theta_ii = m3 - var[:, None] * s          # row i uses var_i
    theta_jj = m3.T - var[None, :] * s        # col j uses var_j
    coef1 = np.sqrt(np.outer(var, 1.0 / np.where(var > 0, var, 1.0)))   # sqrt(v_j/v_i)
    coef2 = np.sqrt(np.outer(1.0 / np.where(var > 0, var, 1.0), var))   # sqrt(v_i/v_j)
    off = (rbar / 2.0) * (coef1 * theta_ii + coef2 * theta_jj)
    np.fill_diagonal(off, 0.0)
    rho_hat = rho_diag + off.sum()

    # gamma: misfit of the target.
    gamma_hat = float(((f - s) ** 2).sum())
    if gamma_hat <= 0:
        return s

    delta = (pi_hat - rho_hat) / gamma_hat
    delta = float(min(1.0, max(0.0, delta)))
    return delta * f + (1.0 - delta) * s


def erc_weights(
    cov: np.ndarray,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    damping: float = 0.5,
) -> np.ndarray | None:
    """Equal-risk-contribution (risk-parity) weights for covariance ``cov``.

    Solves for w (>=0, sum 1) such that every asset contributes equal risk,
    i.e. ``w_i * (cov @ w)_i`` is constant across i. Uses the damped fixed-point
    iteration ``w_i <- (1 / (cov @ w)_i) / sum_j (1 / (cov @ w)_j)`` which
    converges for a positive-definite covariance. Returns ``None`` if it fails
    to produce a finite, positive-weight solution (caller should fall back).
    """
    C = np.atleast_2d(np.asarray(cov, dtype=float))
    n = C.shape[0]
    if n == 0:
        return None
    if n == 1:
        return np.array([1.0])

    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        mrc = C @ w
        mrc = np.where(mrc > 1e-15, mrc, 1e-15)
        target = 1.0 / mrc
        w_new = target / target.sum()
        w_next = (1.0 - damping) * w + damping * w_new
        w_next = w_next / w_next.sum()
        if np.max(np.abs(w_next - w)) < tol:
            w = w_next
            break
        w = w_next

    if not np.all(np.isfinite(w)) or w.sum() <= 0 or np.any(w < 0):
        return None
    return w / w.sum()


def inv_vol_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-volatility weights from a covariance diagonal (ERC's diagonal case)."""
    var = np.clip(np.diag(np.atleast_2d(np.asarray(cov, dtype=float))), 1e-15, None)
    inv = 1.0 / np.sqrt(var)
    return inv / inv.sum()


def risk_contributions(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Fractional risk contribution of each asset (sums to 1)."""
    C = np.atleast_2d(np.asarray(cov, dtype=float))
    w = np.asarray(w, dtype=float)
    port_var = float(w @ C @ w)
    if port_var <= 0:
        return np.full(len(w), 1.0 / len(w))
    rc = w * (C @ w)
    return rc / port_var


def momentum_rank_weights(n: int) -> np.ndarray:
    """Conviction weights for ``n`` names already sorted best→worst momentum.

    Linearly-decaying rank weights (n, n-1, ..., 1) normalised to sum to 1, so
    the strongest momentum name carries the largest tilt. Used as the λ=1 anchor
    of the ERC↔momentum blend.
    """
    if n <= 0:
        return np.array([])
    ranks = np.arange(n, 0, -1, dtype=float)
    return ranks / ranks.sum()
