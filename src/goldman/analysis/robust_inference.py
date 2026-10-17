"""Opt-in OLS and Newey–West inference for the existing monthly regressions.

Pass the same total returns, excess-return factors and risk-free rate used by
``capm`` or ``factor_models``. Coefficients remain OLS; only inference changes.
Alphas and their standard errors stay in monthly decimal units, as in main.
No data are loaded, results written, or analyses run when this module is imported.

HAC uses Bartlett weights, a T/(T-p) finite-sample covariance correction, and
two-sided Student-t p-values with T-p degrees of freedom (an approximation,
not an exact small-sample HAC test). Lag zero is HC1 heteroskedastic inference.
These are individual coefficient tests, not a HAC replacement for the GRS test.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["newey_west_covariance", "robust_capm", "robust_factor_regression"]


def _monthly(data: pd.DataFrame | pd.Series, label: str):
    """Copy onto month starts, rejecting ambiguous or unordered monthly data."""
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError(f"{label} needs a monthly DatetimeIndex or PeriodIndex.")
    result = data.copy()
    if isinstance(result.index, pd.PeriodIndex):
        if result.index.freqstr != "M":
            raise ValueError(f"{label} needs monthly periods.")
        months = result.index
    else:
        months = result.index.to_period("M")
    if months.hasnans or not months.is_unique or not months.is_monotonic_increasing:
        raise ValueError(f"{label} dates must be unique, ordered months without NaT.")
    result.index = months.to_timestamp()
    if isinstance(result, pd.DataFrame):
        if result.shape[1] == 0 or not result.columns.is_unique:
            raise ValueError(f"{label} needs nonempty, unique column names.")
        if not all(isinstance(name, str) and name for name in result.columns):
            raise ValueError(f"{label} column names must be nonempty strings.")
    return result.astype(float)


def _common_inputs(
    returns: pd.DataFrame,
    models: Mapping[str, pd.DataFrame],
    rf: float | pd.Series,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.Series]:
    """Align models on a shared contiguous sample; never fill missing values."""
    if not models:
        raise ValueError("At least one factor model is required.")
    returns = _monthly(returns, "returns")
    aligned = {
        name: _monthly(factors, f"factors for {name}").reindex(returns.index)
        for name, factors in models.items()
    }
    valid = pd.Series(True, index=returns.index)
    for factors in aligned.values():
        valid &= factors.notna().all(axis=1)
    sample = returns.index[valid]
    if sample.empty:
        raise ValueError("No common factor/return observations.")
    if not np.all(np.diff(sample.to_period("M").asi8) == 1):
        raise ValueError("The common sample must be contiguous months for HAC lags.")
    returns = returns.loc[sample]
    aligned = {name: factors.loc[sample] for name, factors in aligned.items()}
    risk_free = (
        _monthly(rf, "rf").reindex(sample)
        if isinstance(rf, pd.Series)
        else pd.Series(float(rf), index=sample)
    )
    for label, values in [("returns", returns), ("rf", risk_free), *aligned.items()]:
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{label} must be finite and cover the common sample.")
    return returns, aligned, risk_free


def _validate_lags(lags: int, n: int) -> None:
    if isinstance(lags, bool) or not isinstance(lags, Integral) or not 0 <= lags < n:
        raise ValueError("lags must be an integer with 0 <= lags < observations.")


def newey_west_covariance(design: np.ndarray, residuals: np.ndarray, lags: int = 3) -> np.ndarray:
    """Bartlett HAC covariance with T/(T-p) correction; include an intercept in X.

    Rows must already represent consecutive, equally spaced observations.
    ``residuals`` is the one-dimensional OLS residual vector for a single asset.
    """
    x = np.asarray(design, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if x.ndim != 2 or residuals.shape != (len(x),):
        raise ValueError("Need a two-dimensional design and one residual per row.")
    n, p = x.shape
    _validate_lags(lags, n)
    if not np.isfinite(x).all() or not np.isfinite(residuals).all():
        raise ValueError("Design and residuals must be finite.")
    if p == 0 or n <= p or np.linalg.matrix_rank(x) != p:
        raise ValueError("Need a full-rank design and observations > parameters.")
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (lags + 1)) * (cross + cross.T)
    bread = np.linalg.inv(x.T @ x)
    covariance = bread @ meat @ bread * n / (n - p)
    return (covariance + covariance.T) / 2


def robust_factor_regression(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    rf: float | pd.Series = 0.0,
    *,
    lags: int = 3,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Return existing-style OLS columns plus HAC inference, one row per asset.

    ``alpha``, ``beta_<factor>``, ``t_alpha``, ``t_<factor>``, ``p_alpha``,
    ``r_squared`` and ``n`` follow ``factor_models.factor_regression`` naming.
    Additional HAC fields end in ``_hac``. Alpha confidence bounds use
    ``alpha_ci_low_hac`` / ``alpha_ci_high_hac``; significance flags are
    unadjusted individual tests at ``significance`` (default 5%).

    Missing factor values at the edges trim the sample as in main; interior
    gaps, nonfinite returns/RF, duplicate months and collinear factors raise.
    Sample dates and the inference convention are recorded in table attributes.
    """
    if not 0 < significance < 1:
        raise ValueError("significance must be between 0 and 1.")
    returns, models, risk_free = _common_inputs(returns, {"model": factors}, rf)
    factors = models["model"]
    if any(name == "alpha" or name.endswith("_hac") for name in factors.columns):
        raise ValueError("Factor names 'alpha' and the '_hac' suffix are reserved for inference.")
    y = returns.sub(risk_free, axis=0).to_numpy()
    n = len(y)
    design = np.column_stack([np.ones(n), factors.to_numpy()])
    p = design.shape[1]
    _validate_lags(lags, n)
    if n <= p or np.linalg.matrix_rank(design) != p:
        raise ValueError("Need a full-rank design and observations > parameters.")
    coefs = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ coefs
    dof = n - p
    sse = (resid**2).sum(axis=0)
    sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    se = np.sqrt(np.diag(np.linalg.inv(design.T @ design))[:, None] * sse / dof)
    hac_se = np.column_stack(
        [
            np.sqrt(np.maximum(np.diag(newey_west_covariance(design, column, lags)), 0))
            for column in resid.T
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ols, t_hac = coefs / se, coefs / hac_se
        r_squared = np.where(sst > 0, 1 - sse / sst, np.nan)
    table = pd.DataFrame(index=returns.columns.copy())
    for position, name in enumerate(["alpha", *factors.columns]):
        coef_name = "alpha" if position == 0 else f"beta_{name}"
        table[coef_name] = coefs[position]
        for suffix, errors, t_values in [("", se, t_ols), ("_hac", hac_se, t_hac)]:
            table[f"se_{name}{suffix}"] = errors[position]
            table[f"t_{name}{suffix}"] = t_values[position]
            table[f"p_{name}{suffix}"] = 2 * stats.t.sf(np.abs(t_values[position]), dof)
    critical = stats.t.ppf(1 - significance / 2, dof)
    table["alpha_ci_low_hac"] = table.alpha - critical * table.se_alpha_hac
    table["alpha_ci_high_hac"] = table.alpha + critical * table.se_alpha_hac
    table["significant_alpha"] = table.p_alpha < significance
    table["significant_alpha_hac"] = table.p_alpha_hac < significance
    table["r_squared"] = r_squared
    table["adjusted_r_squared"] = 1 - (1 - r_squared) * (n - 1) / dof
    table["mean_excess"] = y.mean(axis=0)
    table["n"] = n
    table["residual_df"] = dof
    table["hac_lags"] = lags
    table.index.name = "asset"
    table.attrs.update(
        sample=returns.index.copy(),
        significance=significance,
        hac_kernel="Bartlett",
        hac_correction="T/(T-p)",
        hac_reference="Student-t(T-p)",
        alpha_units="monthly decimal",
    )
    return table


def robust_capm(
    returns: pd.DataFrame,
    market_excess: pd.Series,
    rf: float | pd.Series = 0.0,
    *,
    lags: int = 3,
    significance: float = 0.05,
) -> pd.DataFrame:
    """CAPM adapter using ``beta``, ``se_beta``, ``t_beta`` and ``p_beta`` names.

    Supply the existing market proxy explicitly; this function does not rebuild
    it, change its weights or select a different risk-free series.
    """
    table = robust_factor_regression(
        returns,
        market_excess.to_frame("market"),
        rf,
        lags=lags,
        significance=significance,
    )
    names = {"beta_market": "beta"}
    for prefix in ("se", "t", "p"):
        for suffix in ("", "_hac"):
            names[f"{prefix}_market{suffix}"] = f"{prefix}_beta{suffix}"
    return table.rename(columns=names)
