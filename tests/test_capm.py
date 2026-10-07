"""Tests for the CAPM stage: security characteristic lines, SML and GRS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.capm import (
    empirical_sml,
    excess_returns,
    grs_test,
    plot_sml,
    run_capm_analysis,
    security_characteristic_line,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2010-01-01", periods=n, freq="MS")


def _sim(seed: int, n: int = 120, n_assets: int = 5, alpha: float = 0.0):
    """Return (excess returns, market excess) with known betas and alpha shift."""
    rng = np.random.default_rng(seed)
    dates = _dates(n)
    factor = pd.Series(rng.normal(0.005, 0.04, n), index=dates, name="market_excess")
    betas = np.linspace(0.5, 1.5, n_assets)
    resid = rng.normal(0.0, 0.03, (n, n_assets))
    data = alpha + betas * factor.to_numpy()[:, None] + resid
    columns = [f"a{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=columns), factor


def test_scl_matches_numpy_polyfit():
    rng = np.random.default_rng(0)
    n = 120
    dates = _dates(n)
    market = pd.Series(rng.normal(0.005, 0.04, n), index=dates)
    y = 0.002 + 1.2 * market.to_numpy() + rng.normal(0.0, 0.01, n)
    returns = pd.DataFrame({"a": y}, index=dates)

    table = security_characteristic_line(returns, market, rf=0.0)
    slope, intercept = np.polyfit(market.to_numpy(), y, 1)
    assert table.loc["a", "beta"] == pytest.approx(slope)
    assert table.loc["a", "alpha"] == pytest.approx(intercept)
    assert table.loc["a", "n"] == n
    assert 0.0 <= table.loc["a", "r_squared"] <= 1.0


def test_scl_rf_series_equivalent_to_pre_subtracting():
    excess, market = _sim(1)
    rf = pd.Series(0.002, index=excess.index)
    with_series = security_characteristic_line(excess + 0.002, market, rf=rf)
    pre_subtracted = security_characteristic_line(excess, market, rf=0.0)
    np.testing.assert_allclose(with_series["beta"], pre_subtracted["beta"])
    np.testing.assert_allclose(with_series["alpha"], pre_subtracted["alpha"])


def test_scl_rejects_missing_market_dates():
    excess, market = _sim(2, n=60)
    with pytest.raises(ValueError, match="cover the whole"):
        security_characteristic_line(excess, market.iloc[:30], rf=0.0)


def test_excess_returns_broadcasts_scalar_and_series():
    excess, _ = _sim(3, n=10, n_assets=2)
    assert excess_returns(excess, 0.001).iloc[0, 0] == pytest.approx(excess.iloc[0, 0] - 0.001)
    rf = pd.Series(0.001, index=excess.index)
    pd.testing.assert_frame_equal(excess_returns(excess, rf), excess - 0.001)


def test_empirical_sml_hits_theory_when_alphas_are_zero():
    table = pd.DataFrame(
        {"beta": [0.5, 1.0, 1.5], "mean_excess": [0.003, 0.006, 0.009]},
        index=list("abc"),
    )
    market = pd.Series([0.005] * 10)
    fit = empirical_sml(table, market)
    assert fit.slope == pytest.approx(0.006)
    assert fit.intercept == pytest.approx(0.0, abs=1e-12)
    assert fit.theoretical_slope == pytest.approx(0.005)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.n == 3


def test_grs_reports_wellformed_statistic():
    excess, market = _sim(4, n=120, n_assets=5)
    out = grs_test(excess, market)
    assert out["f_stat"] >= 0.0
    assert 0.0 <= out["p_value"] <= 1.0
    assert out["df1"] == 5
    assert out["df2"] == 120 - 5 - 1
    assert out["n_obs"] == 120


def test_grs_detects_a_shifted_alpha():
    assert grs_test(*_sim(5, alpha=0.05))["f_stat"] > grs_test(*_sim(5, alpha=0.0))["f_stat"]


def test_grs_requires_t_more_than_n_plus_k():
    excess, market = _sim(6, n=6, n_assets=6)
    with pytest.raises(ValueError, match="Not enough observations"):
        grs_test(excess, market)


def test_plot_sml_writes_a_png(tmp_path):
    excess, market = _sim(7, n=120, n_assets=5)
    table = security_characteristic_line(excess, market, rf=0.0)
    fit = empirical_sml(table, market)
    path = plot_sml(table, fit, tmp_path / "sml.png")
    assert path.exists()
    assert path.suffix == ".png"


def test_run_capm_analysis_trims_to_factor_calendar(tmp_path):
    excess, market = _sim(9, n=12, n_assets=4)
    ff3 = pd.DataFrame(
        {"Mkt-RF": market.to_numpy()[:-1]},
        index=pd.PeriodIndex(market.index[:-1], freq="M"),
    )
    with pytest.warns(UserWarning, match="common sample"):
        result = run_capm_analysis(
            excess,
            universe="synthetic",
            method="ff_market",
            ff3=ff3,
            rf=0.0,
            output_dir=tmp_path,
        )
    # One month dropped: 11 observations per asset.
    assert (result.table["n"] == 11).all()


def test_run_capm_analysis_writes_all_exhibits(tmp_path):
    excess, market = _sim(8, n=120, n_assets=6)
    ff3 = pd.DataFrame(
        {"Mkt-RF": market.to_numpy()},
        index=pd.PeriodIndex(market.index, freq="M"),
    )
    result = run_capm_analysis(
        excess,
        universe="synthetic",
        method="ff_market",
        ff3=ff3,
        rf=0.0,
        output_dir=tmp_path,
    )
    for key in ("scl", "sml", "summary"):
        assert result.output_paths[key].exists()
    assert result.fit.n == 6
    assert result.method == "ff_market"
    assert result.table["beta"].notna().all()
