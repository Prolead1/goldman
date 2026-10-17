"""Synthetic checks for coefficient compatibility and HAC inference."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from goldman.analysis.capm import security_characteristic_line
from goldman.analysis.factor_models import factor_regression
from goldman.analysis.robust_inference import (
    newey_west_covariance,
    robust_capm,
    robust_factor_regression,
)


@pytest.fixture
def panel():
    rng = np.random.default_rng(402)
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    factors = pd.DataFrame(rng.normal(0, 0.03, (120, 2)), index=dates, columns=["Mkt-RF", "SMB"])
    rf = pd.Series(np.linspace(0.001, 0.004, 120), index=dates)
    errors = rng.normal(0, 0.015, (120, 3))
    for t in range(1, len(errors)):
        errors[t] += 0.6 * errors[t - 1]
    returns = pd.DataFrame(
        0.002 + factors.to_numpy() @ np.array([[0.7, 1.0, 1.4], [0.2, -0.3, 0.5]]) + errors,
        index=dates,
        columns=["A", "B", "ETF"],
    ).add(rf, axis=0)
    return returns, factors, rf


def test_capm_keeps_all_existing_columns_and_inputs(panel):
    returns, factors, rf = panel
    originals = [obj.copy(deep=True) for obj in panel]
    expected = security_characteristic_line(returns.copy(), factors["Mkt-RF"], rf)
    actual = robust_capm(returns, factors["Mkt-RF"], rf)
    np.testing.assert_allclose(actual[expected.columns], expected, rtol=1e-10, atol=1e-12)
    for actual_input, original in zip(panel, originals, strict=True):
        if isinstance(original, pd.Series):
            pd.testing.assert_series_equal(actual_input, original)
        else:
            pd.testing.assert_frame_equal(actual_input, original)


def test_factor_coefficients_match_main_on_period_indexed_warmup(panel):
    returns, factors, rf = panel
    factors.index = factors.index.to_period("M")
    factors.iloc[:12] = np.nan
    expected = factor_regression(returns, factors.copy(), rf)
    actual = robust_factor_regression(returns, factors, rf)
    np.testing.assert_allclose(actual[expected.columns], expected, rtol=1e-10, atol=1e-12)
    pd.testing.assert_index_equal(actual.attrs["sample"], returns.index[12:])
    assert (actual.n == 108).all()


def test_hc1_intercept_variance_equals_sample_mean_variance():
    y = np.array([1.0, 4.0, 2.0, 5.0, 3.0, 8.0])
    covariance = newey_west_covariance(np.ones((6, 1)), y - y.mean(), lags=0)
    assert covariance[0, 0] == pytest.approx(y.var(ddof=1) / len(y))


def test_bartlett_lag_one_intercept_has_known_covariance():
    # Zero-sum residuals; sum(u^2)=10, sum(u_t*u_{t-1})=4.
    # Bartlett lag-one weight is 1/2, so corrected covariance=(10+4)/(6*5).
    resid = np.array([-2.0, -1.0, 0.0, 0.0, 1.0, 2.0])
    covariance = newey_west_covariance(np.ones((6, 1)), resid, lags=1)
    assert covariance[0, 0] == pytest.approx(14 / 30)


def test_hac_changes_inference_not_estimates_and_ci_matches_p_values(panel):
    returns, factors, rf = panel
    zero = robust_factor_regression(returns, factors, rf, lags=0)
    three = robust_factor_regression(returns, factors, rf, lags=3)
    pd.testing.assert_series_equal(zero.alpha, three.alpha)
    assert not np.allclose(zero.se_alpha_hac, three.se_alpha_hac)
    assert (three.se_alpha_hac > three.se_alpha).all()  # positively correlated errors
    outside = (three.alpha_ci_low_hac > 0) | (three.alpha_ci_high_hac < 0)
    np.testing.assert_array_equal(outside, three.significant_alpha_hac)
    np.testing.assert_allclose(three.p_alpha_hac, 2 * stats.t.sf(abs(three.t_alpha_hac), 117))
    assert three.attrs["alpha_units"] == "monthly decimal"


def test_month_end_rf_and_factors_align_by_month(panel):
    returns, factors, rf = panel
    expected = robust_factor_regression(returns, factors, rf)
    factors.index = factors.index + pd.offsets.MonthEnd(0)
    rf.index = rf.index.to_period("M")
    actual = robust_factor_regression(returns, factors, rf)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("lags", [-1, True, 1.5, 120])
def test_invalid_lags_raise(panel, lags):
    with pytest.raises(ValueError, match="lags"):
        robust_factor_regression(*panel, lags=lags)


@pytest.mark.parametrize("significance", [0, 1, -0.1, np.nan])
def test_invalid_significance_raises(panel, significance):
    with pytest.raises(ValueError, match="significance"):
        robust_factor_regression(*panel, significance=significance)


@pytest.mark.parametrize("kind", ["duplicate", "reverse", "gap", "rf", "returns", "factor_inf"])
def test_invalid_calendars_and_missing_data_raise(panel, kind):
    returns, factors, rf = panel
    if kind == "duplicate":
        returns.index = pd.DatetimeIndex([returns.index[1], *returns.index[1:]])
    elif kind == "reverse":
        returns = returns.iloc[::-1]
    elif kind == "gap":
        factors.iloc[30, 0] = np.nan
    elif kind == "rf":
        rf.iloc[25] = np.nan
    elif kind == "returns":
        returns.iloc[25, 0] = np.inf
    else:
        factors.iloc[25, 0] = np.inf
    with pytest.raises(ValueError):
        robust_factor_regression(returns, factors, rf)


def test_collinear_factors_and_small_samples_raise(panel):
    returns, factors, rf = panel
    factors["duplicate"] = factors["SMB"]
    with pytest.raises(ValueError, match="full-rank"):
        robust_factor_regression(returns, factors, rf)
    with pytest.raises(ValueError, match="full-rank"):
        robust_factor_regression(returns.iloc[:2], factors.iloc[:2, :1], lags=0)


@pytest.mark.parametrize("name", ["alpha", "SMB_hac"])
def test_factor_names_cannot_overwrite_inference_fields(panel, name):
    returns, factors, rf = panel
    factors = factors.rename(columns={"SMB": name})
    with pytest.raises(ValueError, match="reserved"):
        robust_factor_regression(returns, factors, rf)


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "rank"])
def test_covariance_rejects_invalid_inputs(kind):
    x, residuals = np.ones((10, 1)), np.arange(10, dtype=float)
    if kind == "shape":
        residuals = residuals[:, None]
    elif kind == "nonfinite":
        residuals[0] = np.nan
    else:
        x = np.ones((10, 2))
    with pytest.raises(ValueError):
        newey_west_covariance(x, residuals)
