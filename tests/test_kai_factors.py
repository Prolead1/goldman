"""Known-coefficient and matched-model validation for Stage 3."""

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.kai_capm import fit_capm, load_inputs
from goldman.analysis.kai_factors import anomaly_status, fit_factor_model, run


def test_three_factor_coefficients_and_degrees_of_freedom():
    a = np.array([1, -1] * 4)
    b = np.array([1, 1, -1, -1] * 2)
    c = np.array([1] * 4 + [-1] * 4)
    factors = pd.DataFrame({"Mkt-RF": a * .01, "SMB": b * .02, "HML": c * .03})
    error = a * b * .005  # Orthogonal to every regressor, including the intercept.
    excess = pd.DataFrame({"A": .004 + factors @ np.array([1.2, .3, -.4]) + error})
    table, residuals = fit_factor_model(excess, factors)
    r = table.loc["A"]
    assert r.alpha_monthly == pytest.approx(.004)
    assert r.alpha_se_monthly == pytest.approx(.0025)
    assert r.t_alpha == pytest.approx(1.6)
    assert r.residual_df == 4
    np.testing.assert_allclose(r[["loading_Mkt-RF", "loading_SMB", "loading_HML"]]
                              .to_numpy(dtype=float), [1.2, .3, -.4])
    np.testing.assert_allclose(residuals.A, error)


def test_single_factor_matches_existing_capm_including_hac():
    returns, factors = load_inputs()
    excess = returns.sub(factors.RF, axis=0)
    original, residuals = fit_capm(excess, factors["Mkt-RF"])
    new, new_residuals = fit_factor_model(excess, factors[["Mkt-RF"]])
    for column in ["alpha_monthly", "alpha_se_monthly", "t_alpha", "t_alpha_hac", "r_squared"]:
        np.testing.assert_allclose(new[column], original[column])
    np.testing.assert_allclose(new["loading_Mkt-RF"], original.beta)
    pd.testing.assert_frame_equal(new_residuals, residuals)


def test_screening_does_not_confuse_lower_significance_with_smaller_alpha():
    assert anomaly_status(.1, .05, 3, 1) == "attenuated; no longer flagged"
    assert anomaly_status(.1, .2, 3, 1) == "no longer flagged; magnitude not reduced"
    assert anomaly_status(.1, .05, 3, 2.1) == "survives"
    assert anomaly_status(.1, .2, 1, 3) == "newly flagged"
    assert anomaly_status(.1, .05, 1, 1) == "not flagged in either model"


def test_reject_collinear_or_misaligned_factors():
    x = pd.DataFrame({"a": [1., 2., 3., 4., 5.], "b": [2., 4., 6., 8., 10.]})
    y = pd.DataFrame({"asset": [3., 1., 2., 5., 4.]})
    with pytest.raises(ValueError, match="collinear"):
        fit_factor_model(y, x)
    with pytest.raises(ValueError, match="align"):
        fit_factor_model(y, x.iloc[::-1])


def test_full_pack_nested_fit_and_mean_return_identity(tmp_path):
    out = run(tmp_path)
    capm = pd.read_csv(out / "capm_results_matched.csv", index_col=0)
    ff3 = pd.read_csv(out / "ff3_results.csv", index_col=0)
    assert ff3.shape[0] == 21
    assert (ff3.n_months == 118).all()
    assert (ff3.residual_df == 114).all()
    assert (ff3.r_squared >= capm.r_squared - 1e-12).all()
    decomposition = pd.read_csv(out / "mean_return_decomposition.csv", index_col=0)
    np.testing.assert_allclose(decomposition.iloc[:, :-1].sum(axis=1),
                               decomposition.observed_mean_excess_annual, atol=1e-10)
    residuals = pd.read_csv(out / "ff3_residuals.csv", index_col=0)
    factors = pd.read_csv(out / "factors_aligned.csv", index_col=0)
    np.testing.assert_allclose(residuals.mean(), 0, atol=1e-10)
    np.testing.assert_allclose(residuals.T @ factors[["Mkt-RF", "SMB", "HML"]], 0, atol=1e-10)
    assert (out / "stage3_report.md").exists()
