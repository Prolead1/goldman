"""Tests for the no-short and shrinkage robustness layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.portfolio import tangency_weights
from goldman.analysis.robustness import (
    VARIANTS,
    long_only_tangency,
    out_of_sample_sharpe,
    run_robustness_analysis,
    shrink_covariance,
    shrink_means,
    tangency_variants,
    variant_stability,
    variant_weights,
    weight_diagnostics,
)


def _returns(n: int = 120, k: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.006, 0.04, n)
    data = {}
    for i in range(k):
        beta = 0.5 + 0.2 * i
        data[f"a{i}"] = 0.002 + beta * market + rng.normal(0.0, 0.02, n)
    return pd.DataFrame(data, index=pd.date_range("2010-01-01", periods=n, freq="MS"))


def test_shrink_covariance_damps_off_diagonal_only():
    cov = pd.DataFrame([[0.04, 0.02], [0.02, 0.09]], index=list("ab"), columns=list("ab"))
    full = shrink_covariance(cov, 0.0)
    diagonal = shrink_covariance(cov, 1.0)
    pd.testing.assert_frame_equal(full, cov)
    assert diagonal.loc["a", "b"] == 0.0
    assert diagonal.loc["a", "a"] == pytest.approx(0.04)
    assert diagonal.loc["b", "b"] == pytest.approx(0.09)


def test_shrink_covariance_rejects_bad_delta():
    cov = pd.DataFrame(np.eye(2))
    with pytest.raises(ValueError):
        shrink_covariance(cov, 1.5)


def test_shrink_means_moves_toward_the_average():
    means = pd.Series([0.01, 0.03], index=["a", "b"])
    assert shrink_means(means, 1.0).tolist() == pytest.approx([0.02, 0.02])
    pd.testing.assert_series_equal(shrink_means(means, 0.0), means)


def test_long_only_tangency_is_a_valid_long_only_portfolio():
    returns = _returns()
    weights = long_only_tangency(returns.mean() - 0.001, returns.cov())
    assert (weights >= -1e-9).all()
    assert weights.sum() == pytest.approx(1.0)


def test_long_only_concentrates_less_than_unconstrained():
    returns = _returns()
    means = returns.mean() - 0.001
    unconstrained = tangency_weights(means, returns.cov())
    constrained = long_only_tangency(means, returns.cov())
    assert (constrained < -1e-9).sum() == 0
    assert (unconstrained < -1e-9).sum() > 0
    assert constrained.abs().max() <= unconstrained.abs().max() + 1e-9


def test_variant_weights_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        variant_weights(_returns(), 0.001, "magic")


def test_tangency_variants_shape_and_columns():
    returns = _returns()
    frame = tangency_variants(returns, 0.001)
    assert list(frame.columns) == list(VARIANTS)
    assert frame.index.equals(returns.columns)
    assert frame["long_only"].sum() == pytest.approx(1.0)


def test_weight_diagnostics_reports_extremes():
    weights = pd.DataFrame(
        {"unconstrained": [1.5, -0.5], "long_only": [0.6, 0.4]}, index=list("ab")
    )
    diag = weight_diagnostics(weights)
    assert diag.loc["unconstrained", "shorts"] == 1
    assert diag.loc["unconstrained", "gross leverage"] == pytest.approx(2.0)
    assert diag.loc["long_only", "shorts"] == 0
    assert diag.loc["long_only", "max weight"] == pytest.approx(0.6)


def test_variant_stability_and_out_of_sample_sharpe_are_wellformed():
    returns = _returns()
    stability = variant_stability(returns, 0.001)
    sharpes = out_of_sample_sharpe(returns, 0.001)
    assert list(stability.index) == list(VARIANTS)
    assert stability.between(-1.0, 1.0).all()
    assert sharpes.notna().all()


def test_run_robustness_analysis_writes_outputs(tmp_path):
    returns = _returns()
    result = run_robustness_analysis(returns, rf=0.001, output_dir=tmp_path, label="test")
    for key in ("weights", "diagnostics", "chart"):
        assert result.output_paths[key].exists()
    assert set(result.diagnostics.columns) >= {
        "max weight",
        "shorts",
        "gross leverage",
        "half sample weight correlation",
        "out of sample Sharpe",
    }
    assert result.diagnostics.loc["long_only", "shorts"] == 0
