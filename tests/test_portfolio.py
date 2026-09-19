"""Tests for the portfolio math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.portfolio import (
    equal_weights,
    excess_means,
    normalize_weights,
    subsample_tangency,
    tangency_weights,
    weight_correlation,
)


def test_tangency_weights_match_diagonal_closed_form():
    # With a diagonal covariance, w ∝ Sigma^{-1} mu = mu / sigma^2.
    means = pd.Series([0.01, 0.02, 0.04], index=["a", "b", "c"])
    cov = pd.DataFrame(
        np.diag([0.01, 0.02, 0.04]),
        index=means.index,
        columns=means.index,
    )
    weights = tangency_weights(means, cov)

    expected = means.to_numpy() / np.diag(cov.to_numpy())
    expected = expected / expected.sum()
    np.testing.assert_allclose(weights.to_numpy(), expected)
    assert weights.sum() == pytest.approx(1.0)


def test_tangency_weights_sum_to_one():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(60, 5)) * 0.05
    returns = pd.DataFrame(data, columns=list("ABCDE"))
    weights = tangency_weights(excess_means(returns, 0.003), returns.cov())
    assert weights.sum() == pytest.approx(1.0)


def test_equal_weights_sum_to_one():
    weights = equal_weights(["x", "y", "z", "w"])
    assert weights.sum() == pytest.approx(1.0)
    assert (weights == 0.25).all()


def test_normalize_weights_preserves_direction():
    raw = pd.Series([2.0, -1.0, 3.0], index=list("abc"))
    normalized = normalize_weights(raw)
    assert normalized.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(normalized * raw.sum(), raw)


def test_excess_means_with_series_rf():
    returns = pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [0.0, 0.01, 0.0]})
    rf = pd.Series([0.001, 0.001, 0.001], index=returns.index)
    means = excess_means(returns, rf)
    np.testing.assert_allclose(means["a"], returns["a"].mean() - 0.001)


def test_subsample_tangency_shape_and_correlation():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        rng.normal(loc=0.01, scale=0.05, size=(120, 4)),
        columns=list("ABCD"),
    )
    drift = subsample_tangency(returns, rf=0.003)
    assert list(drift.columns) == ["first half", "second half", "full sample"]
    corr = weight_correlation(drift, "first half", "second half")
    assert -1.0 <= corr <= 1.0


def test_subsample_requires_two_splits():
    returns = pd.DataFrame(np.zeros((10, 2)), columns=["a", "b"])
    with pytest.raises(ValueError):
        subsample_tangency(returns, rf=0.0, n_splits=1)