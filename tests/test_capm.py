"""Tests for the Stage 2 CAPM exhibits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.capm import (
    alpha_joint_test,
    capm_regressions,
    capm_table,
    plot_sml,
    run_capm_analysis,
    security_market_line,
    sml_comparison,
)


def _capm_world(seed: int = 0, n_obs: int = 600, n_assets: int = 10, noise: float = 0.02):
    """Returns where the CAPM holds exactly: E[R_i - R_f] = beta_i * premium."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1970-01", periods=n_obs, freq="M")
    market = pd.Series(rng.normal(0.006, 0.045, n_obs), index=idx, name="Mkt-RF")
    betas = np.linspace(0.4, 1.6, n_assets)
    y = np.outer(market, betas) + rng.normal(0.0, noise, (n_obs, n_assets))
    cols = [f"s{i}" for i in range(n_assets)]
    return pd.DataFrame(y, index=idx, columns=cols), market, betas


def test_capm_table_columns_and_beta_recovery():
    excess, market, betas = _capm_world()
    table = capm_table(capm_regressions(excess, market))
    assert list(table.columns) == [
        "beta",
        "t(beta)",
        "alpha (ann.)",
        "t(alpha)",
        "t(alpha) NW",
        "R2",
        "resid vol (ann.)",
    ]
    np.testing.assert_allclose(table["beta"], betas, atol=0.05)
    assert table["resid vol (ann.)"].between(0.05, 0.09).all()  # 0.02 * sqrt(12)


def test_capm_regressions_rejects_multi_column_market():
    excess, market, _ = _capm_world()
    two = pd.concat([market, market.rename("other")], axis=1)
    with pytest.raises(ValueError, match="single series"):
        capm_regressions(excess, two)
    one = market.to_frame()
    assert capm_regressions(excess, one).factors.columns.tolist() == ["Mkt-RF"]


def test_alpha_joint_test_reports_grs_and_counts():
    excess, market, _ = _capm_world()
    joint = alpha_joint_test(capm_regressions(excess, market))
    for key in (
        "GRS statistic",
        "GRS p-value",
        "observed |t|>2",
        "expected by chance",
        "avg residual correlation",
        "avg |residual correlation|",
    ):
        assert key in joint.index
    assert joint["GRS df"] == "F(10, 589)"
    assert joint["GRS p-value"] > 0.001  # the CAPM is true here


def test_sml_matches_theory_when_capm_holds():
    excess, market, _ = _capm_world(seed=1, n_obs=2400, noise=0.01)
    sml = security_market_line(excess, market)
    tests = sml.tests()
    assert list(tests.index) == ["intercept", "slope"]
    assert tests.loc["slope", "theory (ann.)"] == pytest.approx(market.mean() * 12)
    assert abs(tests.loc["intercept", "estimate (ann.)"]) < 0.02
    assert abs(tests.loc["slope", "t vs theory (Shanken)"]) < 3
    assert sml.slope == pytest.approx(tests.loc["slope", "estimate (ann.)"] / 12)


def test_plot_sml_label_modes(tmp_path):
    excess, market, _ = _capm_world()
    sml = security_market_line(excess, market)
    for i, labels in enumerate(["all", "none", ["s0", "s9"], "auto"]):
        path = plot_sml(sml, tmp_path / f"sml_{i}.png", labels=labels)
        assert path.exists() and path.stat().st_size > 0


def test_run_capm_analysis_writes_every_exhibit(tmp_path):
    excess, market, _ = _capm_world()
    result = run_capm_analysis(excess, market, output_dir=tmp_path, prefix="test_")
    assert set(result.output_paths) == {"table", "joint_test", "sml_tests", "sml"}
    for path in result.output_paths.values():
        assert path.exists()
        assert path.name.startswith("test_")
    assert result.grs.df_num == excess.shape[1]
    assert result.table.index.name == "asset"


def test_sml_comparison_one_row_per_case():
    excess, market, _ = _capm_world()
    half = len(excess) // 2
    table = sml_comparison(
        {
            "all": (excess, market),
            "first half": (excess.iloc[:half], market.iloc[:half]),
            "drop s9": (excess.drop(columns="s9"), market),
        }
    )
    assert list(table.index) == ["all", "first half", "drop s9"]
    assert table.loc["first half", "months"] == half
    assert table.loc["drop s9", "assets"] == excess.shape[1] - 1
    assert "t(slope=premium)" in table.columns
