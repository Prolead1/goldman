"""Estimators that tame estimation error in mean–variance inputs.

* :func:`ledoit_wolf_covariance` — Ledoit and Wolf (2004) shrinkage of the
  sample covariance toward a constant-correlation target.
* :func:`capm_prior_means` — the expected excess returns the CAPM implies for a
  given market portfolio: ``pi = beta * market premium``.  Using these as the
  input makes the tangency portfolio *equal* to the market portfolio.
* :func:`shrinkage_intensity` / :func:`shrink_means` — a James–Stein (Jorion
  1986 style) blend of the sample means and that prior, with the weight on the
  prior set by how far the data are from it.

Together they give the "something in between" portfolio of Stage 4: sample
information where it is strong, the market portfolio where it is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "ledoit_wolf_covariance",
    "capm_prior_means",
    "shrinkage_intensity",
    "shrink_means",
]


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Shrink the sample covariance toward a constant-correlation target.

    Implements Ledoit and Wolf (2004), "Honey, I Shrunk the Sample Covariance
    Matrix": ``Sigma = d F + (1 - d) S`` where ``S`` is the sample covariance
    (divided by ``T``), ``F`` keeps the sample variances but sets every
    correlation to the average sample correlation, and the intensity ``d``
    minimises the expected Frobenius loss.  Returns ``(Sigma, d)``.
    """
    x = returns.to_numpy(dtype=float)
    n_obs, n_assets = x.shape
    y = x - x.mean(axis=0)
    sample = y.T @ y / n_obs

    var = np.diag(sample)
    sd = np.sqrt(var)
    corr = sample / np.outer(sd, sd)
    off = ~np.eye(n_assets, dtype=bool)
    r_bar = corr[off].mean()
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    # pi: asymptotic variance of the sample covariance entries.
    y2 = y**2
    pi_mat = (y2.T @ y2) / n_obs - sample**2
    pi_hat = pi_mat.sum()

    # rho: covariance of the target with the sample covariance.
    theta = ((y**3).T @ y) / n_obs - var[:, None] * sample  # theta[i, j]
    ratio = np.outer(1.0 / sd, sd)  # sqrt(s_jj / s_ii)
    rho_off = (r_bar / 2.0) * (ratio * theta + ratio.T * theta.T)
    rho_hat = np.trace(pi_mat) + rho_off[off].sum()

    gamma_hat = ((target - sample) ** 2).sum()
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    intensity = float(np.clip(kappa / n_obs, 0.0, 1.0))

    sigma = intensity * target + (1.0 - intensity) * sample
    cols = returns.columns
    return pd.DataFrame(sigma, index=cols, columns=cols), intensity


def capm_prior_means(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    market_premium: float,
) -> pd.Series:
    """Expected excess returns implied by the CAPM for a given market portfolio.

    ``beta_i = (Sigma w_m)_i / (w_m' Sigma w_m)`` and ``pi_i = beta_i * premium``
    (the reverse optimisation of Black and Litterman).  Because
    ``Sigma^-1 pi`` is proportional to ``w_m``, the tangency portfolio built
    from ``pi`` is exactly the market portfolio.
    """
    w = market_weights.reindex(cov.index).to_numpy(dtype=float)
    sigma = cov.to_numpy(dtype=float)
    beta = sigma @ w / float(w @ sigma @ w)
    return pd.Series(beta * market_premium, index=cov.index, name="capm prior")


def shrinkage_intensity(
    means: pd.Series,
    prior: pd.Series,
    cov: pd.DataFrame,
    n_obs: int,
) -> float:
    """James–Stein weight on the prior: ``(N+2) / ((N+2) + T d' Sigma^-1 d)``.

    ``d`` is the gap between sample and prior means.  Under the null that the
    prior is right, ``T d' Sigma^-1 d`` is roughly chi-squared with ``N``
    degrees of freedom — the same quadratic form that drives the GRS test.  The
    weaker the evidence against the prior, the more the means are shrunk to it.

    ``cov`` should be a well-conditioned estimate (e.g. Ledoit–Wolf): with
    ``T = 60, N = 30`` the inverse of the raw sample covariance is inflated by
    roughly ``(T-1)/(T-N-2)``, which would overstate the evidence and shrink
    too little.
    """
    n_assets = len(means)
    d = (means - prior.reindex(means.index)).to_numpy(dtype=float)
    quad = float(d @ np.linalg.solve(cov.loc[means.index, means.index].to_numpy(), d))
    return float(min(1.0, (n_assets + 2.0) / ((n_assets + 2.0) + n_obs * quad)))


def shrink_means(
    means: pd.Series,
    prior: pd.Series,
    cov: pd.DataFrame,
    n_obs: int,
) -> tuple[pd.Series, float]:
    """Blend sample means with the prior; returns ``(shrunk means, weight on prior)``."""
    phi = shrinkage_intensity(means, prior, cov, n_obs)
    blended = (1.0 - phi) * means + phi * prior.reindex(means.index)
    return blended.rename("shrunk means"), phi
