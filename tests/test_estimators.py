"""Tests for the shrinkage estimators and the constrained portfolio solvers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.estimators import (
    capm_prior_means,
    ledoit_wolf_covariance,
    shrink_means,
    shrinkage_intensity,
)
from goldman.analysis.portfolio import (
    min_variance_weights,
    min_variance_weights_long_only,
    tangency_weights,
    tangency_weights_long_only,
)


def _returns(seed: int = 0, n_obs: int = 60, n_assets: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.008, 0.04, (n_obs, 1))
    y = 0.002 + common * rng.uniform(0.5, 1.5, n_assets) + rng.normal(0, 0.03, (n_obs, n_assets))
    return pd.DataFrame(y, columns=[f"a{i}" for i in range(n_assets)])


def _ledoit_wolf_reference(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Term-by-term transcription of Ledoit and Wolf's covCor.m (MLE sample)."""
    t, n = x.shape
    y = x - x.mean(axis=0)
    sample = y.T @ y / t
    var = np.diag(sample).reshape(-1, 1)
    sqrtvar = np.sqrt(var)
    rho = (np.sum(sample / (sqrtvar @ sqrtvar.T)) - n) / (n * (n - 1))
    prior = rho * (sqrtvar @ sqrtvar.T)
    prior[np.eye(n, dtype=bool)] = var.ravel()
    phi_mat = (y**2).T @ (y**2) / t - sample**2
    phi = phi_mat.sum()
    theta = ((y**3).T @ y) / t - var * sample
    theta[np.eye(n, dtype=bool)] = 0.0
    rho_hat = np.trace(phi_mat) + rho * np.sum(((1.0 / sqrtvar) @ sqrtvar.T) * theta)
    gamma = np.linalg.norm(sample - prior, "fro") ** 2
    delta = max(0.0, min(1.0, (phi - rho_hat) / gamma / t))
    return delta * prior + (1 - delta) * sample, delta


def test_ledoit_wolf_matches_reference_transcription():
    r = _returns()
    sigma, delta = ledoit_wolf_covariance(r)
    ref, ref_delta = _ledoit_wolf_reference(r.to_numpy())
    assert delta == pytest.approx(ref_delta)
    np.testing.assert_allclose(sigma.to_numpy(), ref, atol=1e-15)
    assert 0.0 < delta < 1.0
    np.testing.assert_allclose(np.diag(sigma), r.var(ddof=0))  # variances kept


def test_ledoit_wolf_shrinks_more_with_fewer_observations():
    _, short = ledoit_wolf_covariance(_returns(n_obs=24))
    _, long = ledoit_wolf_covariance(_returns(n_obs=2400))
    assert short > long


def test_capm_prior_makes_market_the_tangency_portfolio():
    r = _returns(1)
    cov = r.cov()
    w_m = pd.Series(np.linspace(1, 2, r.shape[1]), index=r.columns)
    w_m /= w_m.sum()
    prior = capm_prior_means(cov, w_m, market_premium=0.006)
    np.testing.assert_allclose(tangency_weights(prior, cov), w_m)
    # the market's own expected excess return equals the premium (beta = 1)
    assert float(w_m @ prior) == pytest.approx(0.006)


def test_shrinkage_intensity_bounds_and_direction():
    r = _returns(2)
    cov, _ = ledoit_wolf_covariance(r)
    means = r.mean()
    assert shrinkage_intensity(means, means, cov, 60) == pytest.approx(1.0)
    far = means + 0.05
    near = means + 0.0005
    assert shrinkage_intensity(means, far, cov, 60) < shrinkage_intensity(means, near, cov, 60)
    blended, phi = shrink_means(means, far, cov, 60)
    np.testing.assert_allclose(blended, (1 - phi) * means + phi * far)


def test_min_variance_weights_closed_form():
    cov = pd.DataFrame(np.diag([0.01, 0.04]), index=list("ab"), columns=list("ab"))
    w = min_variance_weights(cov)
    assert w["a"] == pytest.approx(0.8)
    lo = min_variance_weights_long_only(cov)
    assert lo["a"] == pytest.approx(0.8, abs=1e-6)


def test_long_only_tangency_on_diagonal_covariance():
    # With diagonal Sigma the long-only max-Sharpe weights are max(mu, 0) / sigma^2.
    idx = list("abcd")
    mu = pd.Series([0.01, -0.005, 0.02, 0.004], index=idx)
    var = np.array([0.02, 0.01, 0.05, 0.01])
    cov = pd.DataFrame(np.diag(var), index=idx, columns=idx)
    w = tangency_weights_long_only(mu, cov)
    expected = np.clip(mu.to_numpy(), 0, None) / var
    np.testing.assert_allclose(w.to_numpy(), expected / expected.sum(), atol=1e-6)
    assert w["b"] == pytest.approx(0.0, abs=1e-9)


def test_long_only_tangency_falls_back_when_nothing_pays():
    idx = list("ab")
    cov = pd.DataFrame(np.diag([0.01, 0.04]), index=idx, columns=idx)
    w = tangency_weights_long_only(pd.Series([-0.01, -0.02], index=idx), cov)
    assert w["a"] == pytest.approx(0.8, abs=1e-6)
