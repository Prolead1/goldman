"""Independent small examples and end-to-end checks for Stage 2."""

import json

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.kai_capm import fit_capm, fit_sml, hac_covariance, run


def test_ols_recovers_known_coefficients_and_residual_variance():
    market = pd.Series([-.04, -.02, 0, .02, .04])
    errors = np.array([.01, -.02, .02, -.02, .01])  # Orthogonal to constant and market.
    y = pd.DataFrame({"A": .003 + 1.2 * market + errors})
    table, residuals = fit_capm(y, market)
    row = table.loc["A"]
    assert row.alpha_monthly == pytest.approx(.003)
    assert row.beta == pytest.approx(1.2)
    expected_se = np.sqrt(np.dot(errors, errors) / 3 / 5)
    assert row.alpha_se_monthly == pytest.approx(expected_se)
    assert row.t_alpha == pytest.approx(.003 / expected_se)
    assert row.alpha_annual == pytest.approx(.036)
    np.testing.assert_allclose(residuals.A, errors)
    expected_r2 = 1 - np.dot(errors, errors) / ((y.A - y.A.mean())**2).sum()
    assert row.r_squared == pytest.approx(expected_r2)


@pytest.mark.parametrize("lags, diagonal", [
    (0, [.0014 / 25 * 5 / 3, 6.4e-7 / .004**2 * 5 / 3]),
    (1, [1 / 75000, 1 / 30]),
])
def test_hac_matches_hand_calculated_covariance(lags, diagonal):
    x = np.column_stack([np.ones(5), [-.04, -.02, 0, .02, .04]])
    errors = np.array([.01, -.02, .02, -.02, .01])
    expected = np.diag(diagonal)
    np.testing.assert_allclose(hac_covariance(x, errors, lags=lags), expected, atol=1e-12)


def test_sml_recovers_known_line_in_annual_units():
    table = pd.DataFrame({"beta": [.5, 1, 1.5], "mean_excess_annual": [.096, .156, .216]})
    result = fit_sml(table, pd.Series([.005, .015]))
    assert result["fitted_intercept_annual"] == pytest.approx(.036)
    assert result["fitted_slope_annual"] == pytest.approx(.12)
    assert result["theoretical_slope_annual"] == pytest.approx(.12)
    assert result["cross_sectional_r_squared"] == pytest.approx(1.)


def test_reject_misaligned_missing_and_constant_market():
    market = pd.Series([-.04, -.02, 0, .02, .04])
    y = pd.DataFrame({"A": [.1, .2, -.1, .04, .01]})
    with pytest.raises(ValueError, match="align"):
        fit_capm(y, market.iloc[::-1])
    with pytest.raises(ValueError, match="finite"):
        fit_capm(y.assign(A=np.nan), market)
    with pytest.raises(ValueError, match="varying market"):
        fit_capm(y, market * 0)


def test_complete_pack_and_ols_pricing_identity(tmp_path):
    out = run(tmp_path)
    table = pd.read_csv(out / "capm_results.csv", index_col=0)
    assert len(table) == 21
    assert (table.n_months == 118).all()
    # With an intercept, the mean excess return is exactly alpha + beta * mean market.
    np.testing.assert_allclose(table.mean_excess_annual,
                               table.alpha_annual + table.capm_excess_annual, atol=1e-10)
    residuals = pd.read_csv(out / "residuals.csv", index_col=0)
    np.testing.assert_allclose(residuals.mean(), 0, atol=1e-10)
    factors = pd.read_csv(out / "factors_aligned.csv", index_col=0)
    np.testing.assert_allclose(residuals.T @ factors["Mkt-RF"], 0, atol=1e-10)
    assessment = json.loads((out / "alpha_assessment.json").read_text())
    assert assessment["formal_joint_test_performed"] is False
    assert assessment["expected_flags_approx_under_null"] == pytest.approx(1.05)
    assert (out / "stage2_report.md").exists()
    assert (out / "sml.png").exists()
