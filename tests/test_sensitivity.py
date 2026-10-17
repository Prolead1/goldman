"""Synthetic checks that comparisons hold factors, dates and assets fixed."""

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.sensitivity import asset_subset_sensitivity, hac_lag_sensitivity


@pytest.fixture
def inputs():
    rng = np.random.default_rng(12)
    dates = pd.date_range("2000-01-01", periods=96, freq="MS")
    factors = pd.DataFrame(rng.normal(0, 0.03, (96, 2)), index=dates, columns=["market", "size"])
    returns = pd.DataFrame(
        factors.market.to_numpy()[:, None] + rng.normal(0, 0.02, (96, 3)),
        index=dates,
        columns=["A", "B", "ETF"],
    )
    later = factors.copy()
    later.iloc[:12] = np.nan
    later.index = later.index.to_period("M")
    return returns, {"CAPM": factors[["market"]], "two_factor": later}


def test_lag_comparison_uses_one_common_sample_and_preserves_coefficients(inputs):
    returns, models = inputs
    result = hac_lag_sensitivity(returns, models)
    assert result.tables.index.names == ["model", "lag", "asset"]
    assert len(result.tables) == 2 * 3 * 3
    assert (result.summary.n == 84).all()
    pd.testing.assert_index_equal(result.sample, returns.index[12:])
    capm = result.tables.loc["CAPM"]
    for lag in (3, 6):
        pd.testing.assert_series_equal(capm.loc[0].alpha, capm.loc[lag].alpha)
        pd.testing.assert_series_equal(capm.loc[0].t_alpha, capm.loc[lag].t_alpha)
    assert (result.summary.significant_alpha_hac <= result.summary.assets).all()
    assert models["two_factor"].iloc[:12].isna().all().all()


def test_subset_comparison_retains_full_baseline_and_shared_asset_results(inputs):
    returns, models = inputs
    result = asset_subset_sensitivity(returns, models, {"stocks_only": ["B", "A"]})
    assert result.tables.index.names == ["subset", "model", "asset"]
    assert result.summary.loc[("full", "CAPM"), "assets"] == 3
    assert result.summary.loc[("stocks_only", "CAPM"), "assets"] == 2
    assert (result.summary.n == 84).all()
    for model in models:
        full = result.tables.loc[("full", model)]
        stocks = result.tables.loc[("stocks_only", model)]
        pd.testing.assert_frame_equal(stocks, full.loc[["B", "A"]])
    pd.testing.assert_index_equal(result.sample, returns.index[12:])


@pytest.mark.parametrize("lags", [(), (0, 0), (0, -1), (0, 84), (True,)])
def test_lag_choices_validated_against_common_sample(inputs, lags):
    with pytest.raises(ValueError):
        hac_lag_sensitivity(*inputs, lags=lags)


@pytest.mark.parametrize(
    "subsets",
    [
        {},
        {"full": ["A"]},
        {"x": []},
        {"x": ["A", "A"]},
        {"x": ["absent"]},
        {"x": "A"},
    ],
)
def test_invalid_subsets_raise(inputs, subsets):
    with pytest.raises(ValueError):
        asset_subset_sensitivity(*inputs, subsets)


def test_no_models_or_common_dates_raise(inputs):
    returns, models = inputs
    with pytest.raises(ValueError, match="At least one"):
        hac_lag_sensitivity(returns, {})
    models["two_factor"].index += 200
    with pytest.raises(ValueError, match="No common"):
        hac_lag_sensitivity(returns, models)


def test_missing_month_in_common_window_is_not_treated_as_consecutive(inputs):
    returns, models = inputs
    models["CAPM"].iloc[30] = np.nan
    with pytest.raises(ValueError, match="contiguous"):
        hac_lag_sensitivity(returns, models)
