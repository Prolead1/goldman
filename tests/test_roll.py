"""Tests for Roll's critique (Stage 4) and the Stage 1 taming exhibit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.portfolio_choice import plot_taming, taming_comparison
from goldman.analysis.roll import (
    ex_post_tangency_proxy,
    max_sharpe,
    plot_proxy_betas,
    proxy_comparison,
    run_roll_analysis,
)


def _world(seed: int = 0, n_obs: int = 240, n_assets: int = 6):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n_obs, freq="M")
    market = pd.Series(rng.normal(0.006, 0.04, n_obs), index=idx, name="Mkt-RF")
    betas = np.linspace(0.6, 1.4, n_assets)
    y = np.outer(market, betas) + rng.normal(0.001, 0.03, (n_obs, n_assets))
    excess = pd.DataFrame(y, index=idx, columns=[f"a{i}" for i in range(n_assets)])
    return excess, market


def test_ex_post_tangency_proxy_prices_everything():
    excess, _ = _world()
    fit_alpha = proxy_comparison(excess, {"tan": ex_post_tangency_proxy(excess)}).fits["tan"]
    assert fit_alpha.alpha.abs().max() < 1e-12


def test_grs_is_a_function_of_sharpe_ratios():
    excess, market = _world(1)
    res = proxy_comparison(excess, {"mkt": market})
    n_obs, n_assets = excess.shape
    sr_p = market.mean() / market.std(ddof=0)
    sr_star = max_sharpe(pd.concat([excess, market], axis=1))
    expected = (n_obs - n_assets - 1) / n_assets * (sr_star**2 - sr_p**2) / (1 + sr_p**2)
    assert res.summary.loc["mkt", "GRS"] == pytest.approx(expected)


def test_proxy_comparison_columns_and_baseline():
    excess, market = _world(2)
    noisy = market + np.random.default_rng(3).normal(0, 0.01, len(market))
    res = proxy_comparison(
        excess,
        {"base": market, "noisy": noisy, "tan": ex_post_tangency_proxy(excess)},
        highlight=["a0"],
    )
    assert res.summary.loc["base", "mean |beta change| vs baseline"] == 0.0
    assert res.summary.loc["noisy", "mean |beta change| vs baseline"] > 0.0
    assert res.summary.loc["base", "corr(alpha, baseline alpha)"] == pytest.approx(1.0)
    assert np.isnan(res.summary.loc["tan", "corr(alpha, baseline alpha)"])
    assert "alpha a0 (ann.)" in res.summary.columns
    assert list(res.betas.columns) == ["base", "noisy", "tan"]


def test_proxy_must_cover_every_month():
    excess, market = _world(4)
    with pytest.raises(ValueError, match="cover"):
        proxy_comparison(excess, {"short": market.iloc[1:]})


def test_run_roll_analysis_writes_exhibits(tmp_path):
    excess, market = _world(5)
    res = run_roll_analysis(
        excess, {"base": market, "ew": excess.mean(axis=1)}, output_dir=tmp_path, prefix="t_"
    )
    assert set(res.output_paths) == {"summary", "betas", "alphas", "beta_chart", "alpha_chart"}
    for path in res.output_paths.values():
        assert path.exists() and path.name.startswith("t_")
    too_many = pd.DataFrame({f"p{i}": res.betas["base"] for i in range(6)})
    with pytest.raises(ValueError):
        plot_proxy_betas(too_many, tmp_path / "x.png")


def test_taming_comparison_tames_the_weights(tmp_path):
    rng = np.random.default_rng(6)
    excess, market = _world(6, n_obs=120, n_assets=10)
    caps = pd.Series(rng.uniform(1, 5, 10), index=excess.columns)
    taming = taming_comparison(excess, caps, prior_premium=0.005)
    table = taming.table
    assert table.loc["tangency (long-only)", "short positions"] == 0
    assert table.loc["tangency (long-only)", "gross exposure"] == pytest.approx(1.0)
    assert (
        table.loc["tangency (plug-in)", "half-sample L1 distance"]
        >= table.loc["tangency (shrinkage, long-only)", "half-sample L1 distance"]
    )
    assert 0.0 < table.loc["tangency (shrinkage)", "weight on CAPM prior"] <= 1.0
    np.testing.assert_allclose(taming.weights.sum(), 1.0, atol=1e-6)
    assert plot_taming(taming, tmp_path / "taming.png").exists()
