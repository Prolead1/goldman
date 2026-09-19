"""Portfolio-construction math.

Implements the mean–variance objects used across the analyses: the tangency
portfolio ``w ∝ Sigma^{-1} mu_e``, equal/value weighting, and the subsample
re-estimation that makes estimation error visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "excess_means",
    "tangency_weights",
    "normalize_weights",
    "equal_weights",
    "subsample_tangency",
    "weight_correlation",
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


def _tangency_for(returns: pd.DataFrame, rf: float | pd.Series) -> pd.Series:
    means = excess_means(returns, rf)
    return tangency_weights(means, returns.cov())


def _split_label(index: int, n_splits: int) -> str:
    words = ["first", "second", "third", "fourth", "fifth"]
    if index < len(words):
        return f"{words[index]} half" if n_splits == 2 else f"{words[index]} split"
    return f"split {index + 1}"


def weight_correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    """Correlation between two weight columns — the 0.16-style headline."""
    return float(frame[left].corr(frame[right]))