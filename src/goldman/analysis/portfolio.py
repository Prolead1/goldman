"""Portfolio-construction math.

Implements the mean–variance objects used across the analyses: the tangency
portfolio ``w ∝ Sigma^{-1} mu_e`` (unconstrained and long-only), the minimum
variance portfolio, equal/value weighting, and the subsample re-estimation
that makes estimation error visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "excess_means",
    "excess_returns",
    "tangency_weights",
    "tangency_weights_long_only",
    "min_variance_weights",
    "min_variance_weights_long_only",
    "normalize_weights",
    "equal_weights",
    "subsample_tangency",
    "weight_correlation",
    "two_fund_allocation",
]


def excess_means(returns: pd.DataFrame, rf: float | pd.Series) -> pd.Series:
    """Mean monthly excess return for each asset.

    ``rf`` may be a scalar or a per-period ``Series`` aligned to ``returns``.
    """
    return returns.mean() - _as_mean_rf(returns, rf)


def _as_mean_rf(returns: pd.DataFrame, rf: float | pd.Series) -> float | pd.Series:
    if isinstance(rf, pd.Series):
        return rf.reindex(returns.index).mean()
    return rf


def tangency_weights(
    means: pd.Series,
    cov: pd.DataFrame | np.ndarray,
) -> pd.Series:
    """Solve ``w ∝ Sigma^{-1} mu_e`` and normalise the weights to sum to 1.

    The covariance argument is expected to match the order of ``means``.
    """
    cov_array = np.asarray(cov, dtype=float)
    means_array = np.asarray(means, dtype=float)
    raw = np.linalg.solve(cov_array, means_array)
    weights = raw / raw.sum()
    return pd.Series(weights, index=means.index, name="tangency")


def _min_quadratic_long_only(
    cov: np.ndarray,
    a: np.ndarray,
    x0: np.ndarray,
) -> np.ndarray:
    """Solve ``min x' Sigma x`` s.t. ``a' x = 1, x >= 0`` (a small convex QP)."""
    result = minimize(
        lambda x: x @ cov @ x,
        x0,
        jac=lambda x: 2.0 * cov @ x,
        bounds=[(0.0, None)] * len(x0),
        constraints=[{"type": "eq", "fun": lambda x: a @ x - 1.0, "jac": lambda x: a}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not result.success:  # pragma: no cover - SLSQP on a convex QP
        raise RuntimeError(f"long-only optimisation failed: {result.message}")
    return np.clip(result.x, 0.0, None)


def min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """The global minimum-variance portfolio ``w ∝ Sigma^{-1} 1``."""
    raw = np.linalg.solve(cov.to_numpy(dtype=float), np.ones(len(cov)))
    return pd.Series(raw / raw.sum(), index=cov.index, name="min variance")


def min_variance_weights_long_only(cov: pd.DataFrame) -> pd.Series:
    """Minimum variance with no short sales (fully invested)."""
    sigma = cov.to_numpy(dtype=float)
    n = len(cov)
    x = _min_quadratic_long_only(sigma, np.ones(n), np.full(n, 1.0 / n))
    return pd.Series(x / x.sum(), index=cov.index, name="min variance (long-only)")


def tangency_weights_long_only(means: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Maximum-Sharpe portfolio with no short sales.

    Solved as ``min y' Sigma y`` s.t. ``mu_e' y = 1, y >= 0`` and ``w = y / 1'y``
    (the standard reformulation of the max-Sharpe problem).  If no asset has a
    positive expected excess return no long-only portfolio has a positive
    Sharpe ratio, and the long-only minimum-variance portfolio is returned.
    """
    mu = means.reindex(cov.index).to_numpy(dtype=float)
    if not (mu > 0).any():
        return min_variance_weights_long_only(cov).rename("tangency (long-only)")
    sigma = cov.to_numpy(dtype=float)
    pos = np.clip(mu, 0.0, None)
    x0 = pos / (mu @ pos)
    y = _min_quadratic_long_only(sigma, mu, x0)
    return pd.Series(y / y.sum(), index=cov.index, name="tangency (long-only)")


def normalize_weights(weights: pd.Series) -> pd.Series:
    """Scale a weight vector so its entries sum to one."""
    return weights / weights.sum()


def equal_weights(assets: pd.Index | list[str]) -> pd.Series:
    """The 1/N portfolio over ``assets``."""
    n = len(assets)
    return pd.Series(np.full(n, 1.0 / n), index=assets, name="equal")


def subsample_tangency(
    returns: pd.DataFrame,
    rf: float | pd.Series,
    n_splits: int = 2,
) -> pd.DataFrame:
    """Re-estimate the tangency portfolio on consecutive subsamples.

    Returns a frame with one column per split (``first half``, ``second half``,
    …) plus a ``full sample`` column, indexed by asset.  The instability of
    these columns is the estimation-error story: the estimated weights are
    unstable across subsamples.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    out: dict[str, pd.Series] = {}
    edges = np.linspace(0, len(returns), n_splits + 1, dtype=int)
    for i in range(n_splits):
        sub = returns.iloc[edges[i] : edges[i + 1]]
        label = _split_label(i, n_splits)
        out[label] = _tangency_for(sub, rf)

    out["full sample"] = _tangency_for(returns, rf)
    return pd.DataFrame(out)


def excess_returns(returns: pd.DataFrame, rf: float | pd.Series) -> pd.DataFrame:
    """Returns minus the risk-free rate (a scalar or a per-period series)."""
    if isinstance(rf, pd.Series):
        return returns.sub(rf.reindex(returns.index), axis=0)
    return returns - rf


def _tangency_for(returns: pd.DataFrame, rf: float | pd.Series) -> pd.Series:
    """Tangency weights from the mean and covariance of *excess* returns."""
    excess = excess_returns(returns, rf)
    return tangency_weights(excess.mean(), excess.cov())


def _split_label(index: int, n_splits: int) -> str:
    words = ["first", "second", "third", "fourth", "fifth"]
    if index < len(words):
        return f"{words[index]} half" if n_splits == 2 else f"{words[index]} split"
    return f"split {index + 1}"


def weight_correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    """Correlation between two weight columns — the 0.16-style headline."""
    return float(frame[left].corr(frame[right]))

def two_fund_allocation(
    market_excess: pd.Series,
    risk_aversions: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0),
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """The Mutual-Fund Theorem in numbers: how much of the index to hold.

    Every mean–variance investor holds the same risky fund (the market) and
    splits wealth between it and T-bills: ``y* = premium / (gamma * sigma^2)``
    for utility ``E[r] - gamma/2 * var``.  The premium is estimated with
    error, so the share is also shown at the premium plus or minus two
    standard errors.  Inputs are monthly excess returns; output is annual.
    """
    n = int(market_excess.notna().sum())
    mean = float(market_excess.mean()) * periods_per_year
    var = float(market_excess.var()) * periods_per_year
    se = float(market_excess.std()) / np.sqrt(n) * periods_per_year
    rows = {}
    for gamma in risk_aversions:
        rows[gamma] = {
            "risky share y*": mean / (gamma * var),
            "y* if premium - 2se": (mean - 2 * se) / (gamma * var),
            "y* if premium + 2se": (mean + 2 * se) / (gamma * var),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "risk aversion"
    out.attrs.update(
        {"premium (ann.)": mean, "vol (ann.)": np.sqrt(var), "premium se (ann.)": se, "months": n}
    )
    return out
