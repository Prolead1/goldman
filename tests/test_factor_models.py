"""Tests for the multi-factor regressions and the alpha comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goldman.analysis.capm import grs_test
from goldman.analysis.factor_models import (
    alpha_comparison,
    factor_regression,
    model_summary,
    run_factor_analysis,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2010-01-01", periods=n, freq="MS")


def _sim(seed: int = 0, n: int = 120, n_assets: int = 5, start_zero: bool = False):
    """Return (returns, factor frame) with known betas and alphas."""
    rng = np.random.default_rng(seed)
    dates = _dates(n)
    market = rng.normal(0.005, 0.04, n)
    size = rng.normal(0.0, 0.02, n)
    factors = pd.DataFrame({"Mkt-RF": market, "SMB": size}, index=dates)
    if start_zero:
        factors.iloc[:5] = np.nan  # a factor with a shorter history
    alpha = 0.001
    data = {}
    for i in range(n_assets):
        beta = 0.5 + 0.2 * i
        data[f"a{i}"] = (
            alpha + beta * market + 0.3 * size + rng.normal(0.0, 0.005, n)
        )
    return pd.DataFrame(data, index=dates), factors


def test_factor_regression_recovers_known_coefficients():
    returns, factors = _sim(seed=1)
    table = factor_regression(returns, factors, rf=0.0)
    assert list(table.index) == list(returns.columns)
    assert table["alpha"].sub(0.001).abs().max() < 0.002
    assert (
        table["beta_Mkt-RF"]
        .sub(pd.Series(np.linspace(0.5, 1.3, 5), index=table.index))
        .abs()
        .max()
        < 0.03
    )
    assert table["beta_SMB"].sub(0.3).abs().max() < 0.05
    assert table["r_squared"].between(0.0, 1.0).all()
    assert (table["n"] == 120).all()
    assert table["p_alpha"].between(0.0, 1.0).all()


def test_factor_regression_accepts_period_indexed_factors():
    returns, factors = _sim(seed=2, n=60, n_assets=3)
    period = factors.copy()
    period.index = pd.PeriodIndex(factors.index, freq="M")
    onsets = factor_regression(returns, period, rf=0.0)
    datetimes = factor_regression(returns, factors, rf=0.0)
    pd.testing.assert_frame_equal(onsets, datetimes)


def test_factor_regression_drops_months_with_missing_factors():
    returns, factors = _sim(seed=3, start_zero=True)
    table = factor_regression(returns, factors, rf=0.0)
    assert (table["n"] == 115).all()


def test_factor_regression_rejects_no_overlap():
    returns, factors = _sim(seed=4, n=40, n_assets=3)
    shifted = factors.copy()
    shifted.index = shifted.index + pd.DateOffset(years=5)
    try:
        factor_regression(returns, shifted, rf=0.0)
    except ValueError as exc:
        assert "overlapping" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("expected a ValueError")


def test_alpha_comparison_and_summary():
    returns, factors = _sim(seed=5)
    market_only = factors[["Mkt-RF"]]
    tables = {
        "CAPM": factor_regression(returns, market_only, rf=0.0),
        "Two-factor": factor_regression(returns, factors, rf=0.0),
    }
    alphas = alpha_comparison(tables)
    assert list(alphas.columns) == ["CAPM", "Two-factor"]
    assert alphas.index.equals(returns.columns)

    grs = {
        name: grs_test(returns, model)
        for name, model in {"CAPM": market_only, "Two-factor": factors}.items()
    }
    summary = model_summary(tables, grs)
    assert list(summary.index) == ["CAPM", "Two-factor"]
    assert summary.loc["CAPM", "assets"] == len(returns.columns)
    assert summary["GRS p"].between(0.0, 1.0).all()


def test_grs_accepts_multiple_factors():
    returns, factors = _sim(seed=6)
    out = grs_test(returns, factors)
    assert out["df1"] == len(returns.columns)
    assert out["df2"] == 120 - len(returns.columns) - 2
    assert out["f_stat"] >= 0.0


def test_run_factor_analysis_uses_common_sample_and_writes_outputs(tmp_path):
    returns, factors = _sim(seed=7, start_zero=True)
    market_only = factors[["Mkt-RF"]]
    models = {"CAPM": market_only, "Two-factor": factors}
    result = run_factor_analysis(
        returns, universe="synthetic", models=models, rf=0.0, output_dir=tmp_path
    )
    assert len(result.sample) == 115
    assert (result.tables["CAPM"]["n"] == 115).all()
    assert result.models == ("CAPM", "Two-factor")
    for key in ("CAPM scl", "Two-factor scl", "alphas", "summary", "chart"):
        assert result.output_paths[key].exists()


def test_run_factor_analysis_rejects_models_without_overlap():
    returns, factors = _sim(seed=8, n=40, n_assets=3)
    shifted = factors.copy()
    shifted.index = shifted.index + pd.DateOffset(years=5)
    try:
        run_factor_analysis(
            returns,
            universe="synthetic",
            models={"CAPM": factors[["Mkt-RF"]], "later": shifted},
            rf=0.0,
        )
    except ValueError as exc:
        assert "common observations" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("expected a ValueError")
