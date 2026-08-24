"""Unit tests for the covariance-aware weighting module (ERC + Ledoit-Wolf)."""
from __future__ import annotations

import numpy as np
import pytest

from backtest._weighting import (
    erc_weights,
    inv_vol_weights,
    ledoit_wolf_constant_corr,
    momentum_rank_weights,
    risk_contributions,
    sample_covariance,
)


@pytest.mark.unit
def test_erc_reduces_to_inv_vol_on_diagonal_cov():
    """Under zero correlation, ERC must collapse to inverse-vol (the thesis)."""
    cov = np.diag([0.04, 0.01, 0.09])  # vols 0.2, 0.1, 0.3
    erc = erc_weights(cov)
    iv = inv_vol_weights(cov)
    assert erc is not None
    np.testing.assert_allclose(erc, iv, atol=1e-4)
    # Lowest-vol asset gets the largest weight.
    assert erc[1] > erc[0] > erc[2]


@pytest.mark.unit
def test_erc_equalises_risk_contributions_with_correlation():
    cov = np.array(
        [
            [0.04, 0.018, 0.006],
            [0.018, 0.09, 0.012],
            [0.006, 0.012, 0.0225],
        ]
    )
    w = erc_weights(cov)
    assert w is not None
    rc = risk_contributions(cov, w)
    # All three risk contributions equal → each ≈ 1/3.
    np.testing.assert_allclose(rc, np.full(3, 1 / 3), atol=1e-3)


@pytest.mark.unit
def test_erc_weights_sum_to_one_and_nonnegative():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((250, 5))
    cov = np.cov(A, rowvar=False)
    w = erc_weights(cov)
    assert w is not None
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w >= 0)


@pytest.mark.unit
def test_ledoit_wolf_is_symmetric_psd_and_between_sample_and_target():
    rng = np.random.default_rng(42)
    # Correlated 5-asset returns, short sample (noisy covariance).
    factor = rng.standard_normal((60, 1))
    noise = rng.standard_normal((60, 5))
    X = 0.01 * (factor + 0.5 * noise)
    lw = ledoit_wolf_constant_corr(X)
    # Symmetric.
    np.testing.assert_allclose(lw, lw.T, atol=1e-12)
    # Positive semi-definite (eigenvalues >= 0).
    eig = np.linalg.eigvalsh(lw)
    assert eig.min() > -1e-10
    # Diagonal (variances) preserved. LW uses the 1/T (MLE) normalisation
    # internally, so compare against the ddof=0 variances.
    np.testing.assert_allclose(np.diag(lw), np.var(X, axis=0, ddof=0), rtol=1e-6)


@pytest.mark.unit
def test_ledoit_wolf_pulls_correlations_toward_average():
    """Shrinkage compresses the spread of off-diagonal correlations."""
    rng = np.random.default_rng(7)
    X = 0.01 * rng.standard_normal((40, 6))  # very short sample → strong shrinkage
    s = sample_covariance(X)
    lw = ledoit_wolf_constant_corr(X)

    def corr_offdiag(m):
        d = np.sqrt(np.diag(m))
        c = m / np.outer(d, d)
        return c[np.triu_indices(len(d), k=1)]

    assert corr_offdiag(lw).std() < corr_offdiag(s).std()


@pytest.mark.unit
def test_momentum_rank_weights_decreasing_and_normalised():
    w = momentum_rank_weights(5)
    assert abs(w.sum() - 1.0) < 1e-12
    assert all(w[i] > w[i + 1] for i in range(len(w) - 1))
