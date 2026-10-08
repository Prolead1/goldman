"""Econometrics shared by the CAPM and factor-model stages.

* :func:`time_series_regression` — one OLS per asset of excess returns on a set
  of factor excess returns, with homoskedastic and Newey–West t-statistics.
* :func:`grs_test` — the Gibbons–Ross–Shanken (1989) test that every alpha is
  zero, which accounts for correlated residuals across assets.
* :func:`chance_exceedances` — how many ``|t| > 2`` alphas pure chance would
  produce, the informal companion to the GRS test.
* :func:`fama_macbeth` — cross-sectional regressions of returns on betas, one
  per month, with Fama–MacBeth and Shanken (1992) errors-in-variables
  corrected standard errors.

Everything is plain numpy so the formulas stay visible; scipy supplies only the
F, t and binomial distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "TimeSeriesFit",
    "GRSResult",
    "FamaMacBethResult",
    "newey_west_lags",
    "time_series_regression",
    "grs_test",
    "chance_exceedances",
    "fama_macbeth",
]


# ------------------------------------------------------- time-series OLS
@dataclass
class TimeSeriesFit:
    """Per-asset OLS fits of excess returns on factors.

    ``params``, ``tstats`` and ``tstats_hac`` are asset-by-coefficient frames
    whose first column is ``alpha`` followed by one column per factor.  Values
    are monthly (not annualised).
    """

    params: pd.DataFrame
    tstats: pd.DataFrame
    tstats_hac: pd.DataFrame
    r2: pd.Series
    resid: pd.DataFrame
    factors: pd.DataFrame
    hac_lags: int

    @property
    def alpha(self) -> pd.Series:
        return self.params["alpha"]

    @property
    def n_obs(self) -> int:
        return len(self.resid)

    @property
    def n_assets(self) -> int:
        return self.resid.shape[1]

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom of each regression: ``T - K - 1``."""
        return self.n_obs - self.factors.shape[1] - 1


def newey_west_lags(n_obs: int) -> int:
    """Rule-of-thumb Newey–West lag length ``floor(4 (T/100)^(2/9))``."""
    return int(np.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0)))


def _hac_cov(x: np.ndarray, e: np.ndarray, xtx_inv: np.ndarray, lags: int) -> np.ndarray:
    """Newey–West (Bartlett kernel) covariance of OLS coefficients for one asset."""
    scores = x * e[:, None]  # T x (K+1), row t is x_t * e_t
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    return xtx_inv @ meat @ xtx_inv


def time_series_regression(
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    hac_lags: int | None = None,
) -> TimeSeriesFit:
    """Regress each asset's excess return on the factors, with an intercept.

    ``R_i,t - R_f,t = alpha_i + b_i' f_t + e_i,t`` for every column of
    ``excess_returns``.  ``factors`` must hold *only* the regressors (excess
    returns or long–short spreads) on the same index.  ``hac_lags`` defaults to
    :func:`newey_west_lags`.
    """
    if not excess_returns.index.equals(factors.index):
        raise ValueError("excess_returns and factors must share the same index.")
    if excess_returns.isna().any().any() or factors.isna().any().any():
        raise ValueError("Regression inputs contain missing values; align and clean first.")

    y = excess_returns.to_numpy(dtype=float)
    f = factors.to_numpy(dtype=float)
    n_obs, n_factors = f.shape
    if n_obs <= n_factors + 1:
        raise ValueError("Not enough observations for the number of factors.")
    lags = newey_west_lags(n_obs) if hac_lags is None else int(hac_lags)

    x = np.column_stack([np.ones(n_obs), f])
    xtx_inv = np.linalg.inv(x.T @ x)
    coefs = xtx_inv @ x.T @ y  # (K+1) x N
    resid = y - x @ coefs
    dof = n_obs - n_factors - 1

    sigma2 = (resid**2).sum(axis=0) / dof
    se = np.sqrt(np.outer(np.diag(xtx_inv), sigma2))  # (K+1) x N
    se_hac = np.column_stack(
        [np.sqrt(np.diag(_hac_cov(x, resid[:, i], xtx_inv, lags))) for i in range(y.shape[1])]
    )

    centred = y - y.mean(axis=0)
    r2 = 1.0 - (resid**2).sum(axis=0) / (centred**2).sum(axis=0)

    names = ["alpha", *factors.columns]
    assets = excess_returns.columns

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values.T, index=assets, columns=names)

    return TimeSeriesFit(
        params=frame(coefs),
        tstats=frame(coefs / se),
        tstats_hac=frame(coefs / se_hac),
        r2=pd.Series(r2, index=assets, name="R2"),
        resid=pd.DataFrame(resid, index=excess_returns.index, columns=assets),
        factors=factors.copy(),
        hac_lags=lags,
    )


# ------------------------------------------------------------- GRS test
@dataclass
class GRSResult:
    """Gibbons–Ross–Shanken statistic, its F distribution and p-value."""

    stat: float
    p_value: float
    df_num: int
    df_den: int


def grs_test(fit: TimeSeriesFit) -> GRSResult:
    """Test ``H0: all alphas are zero`` jointly (Gibbons, Ross and Shanken 1989).

    ``GRS = (T - N - K) / N * a' S^-1 a / (1 + m' W^-1 m)`` with ``S`` the
    residual covariance and ``W`` the factor covariance (both maximum
    likelihood, divided by ``T``) and ``m`` the factor means.  Under normal
    errors it is exactly ``F(N, T - N - K)``.  Unlike counting individual
    t-statistics, it accounts for the correlation between the residuals.
    """
    n_obs, n_assets = fit.resid.shape
    n_factors = fit.factors.shape[1]
    df_den = n_obs - n_assets - n_factors
    if df_den <= 0:
        raise ValueError("GRS needs T > N + K: too many assets for the sample length.")

    resid = fit.resid.to_numpy()
    sigma = resid.T @ resid / n_obs
    f = fit.factors.to_numpy()
    mu = f.mean(axis=0)
    omega = np.atleast_2d(np.cov(f, rowvar=False, bias=True))
    alpha = fit.alpha.to_numpy()

    quad_alpha = float(alpha @ np.linalg.solve(sigma, alpha))
    quad_factor = float(mu @ np.linalg.solve(omega, mu))
    stat = (df_den / n_assets) * quad_alpha / (1.0 + quad_factor)
    p_value = float(stats.f.sf(stat, n_assets, df_den))
    return GRSResult(stat=float(stat), p_value=p_value, df_num=n_assets, df_den=df_den)


def chance_exceedances(
    tstats: pd.Series,
    df: int,
    threshold: float = 2.0,
) -> dict[str, float]:
    """How many ``|t| > threshold`` would chance alone produce?

    Under ``H0: alpha = 0`` each t-statistic is Student-t with ``df`` degrees of
    freedom, so each exceeds the threshold with probability ``p``.  If the tests
    were independent the count would be ``Binomial(N, p)``; residuals are in
    fact positively correlated, which makes clusters of large t-stats more
    likely than the binomial tail suggests — the GRS test handles that properly.
    """
    n = int(tstats.notna().sum())
    observed = int((tstats.abs() > threshold).sum())
    p_single = float(2.0 * stats.t.sf(threshold, df))
    min_p = float(2.0 * stats.t.sf(tstats.abs().max(), df))
    return {
        "n tests": n,
        "observed |t|>2": observed,
        "P(|t|>2) per test": p_single,
        "expected by chance": n * p_single,
        "P(at least observed)": float(stats.binom.sf(observed - 1, n, p_single)),
        "Bonferroni p (largest |t|)": min(1.0, n * min_p),
    }


# --------------------------------------------------------- Fama–MacBeth
@dataclass
class FamaMacBethResult:
    """Average cross-sectional coefficients and their standard errors.

    ``gamma`` is indexed ``const`` then one entry per beta column; values are
    monthly.  ``se_shanken`` adds the Shanken (1992) correction for betas being
    estimated rather than known.
    """

    gamma: pd.Series
    se: pd.Series
    se_shanken: pd.Series
    gamma_t: pd.DataFrame
    r2_cross_section: float

    @property
    def tstats(self) -> pd.Series:
        return self.gamma / self.se

    @property
    def tstats_shanken(self) -> pd.Series:
        return self.gamma / self.se_shanken


def fama_macbeth(
    excess_returns: pd.DataFrame,
    betas: pd.DataFrame | pd.Series,
    factors: pd.DataFrame | pd.Series | None = None,
) -> FamaMacBethResult:
    """Fama–MacBeth (1973) cross-sectional regressions on fixed betas.

    Each month ``t`` regresses the cross-section of excess returns on a
    constant and the betas: ``R_i,t - R_f,t = g0_t + g1_t' b_i + u_i,t``.  The
    estimates are the time-series means of ``g_t`` and their standard errors
    come from the time-series variation of ``g_t``, which is robust to
    cross-sectional correlation of returns.  With full-sample betas the mean
    ``g`` equals the OLS fit of average returns on betas.

    ``factors`` (the time series behind the betas) enables the Shanken
    correction: ``V = (1 + c) (V_FM - S/T) + S/T`` with
    ``c = lambda' Sigma_f^-1 lambda``, where ``S`` is the factor covariance
    placed in the slope block.
    """
    beta_frame = betas.to_frame() if isinstance(betas, pd.Series) else betas
    beta_frame = beta_frame.loc[excess_returns.columns]
    names = ["const", *beta_frame.columns]

    x = np.column_stack([np.ones(len(beta_frame)), beta_frame.to_numpy(dtype=float)])
    y = excess_returns.to_numpy(dtype=float)  # T x N
    gamma_t = np.linalg.lstsq(x, y.T, rcond=None)[0].T  # T x (K+1)
    n_obs = gamma_t.shape[0]

    gamma = gamma_t.mean(axis=0)
    cov_fm = np.atleast_2d(np.cov(gamma_t, rowvar=False, ddof=1)) / n_obs
    se = np.sqrt(np.diag(cov_fm))

    if factors is not None:
        f = factors.to_frame() if isinstance(factors, pd.Series) else factors
        sigma_f = np.atleast_2d(np.cov(f.to_numpy(dtype=float), rowvar=False, ddof=1))
        lam = gamma[1:]
        c = float(lam @ np.linalg.solve(sigma_f, lam))
        bordered = np.zeros_like(cov_fm)
        bordered[1:, 1:] = sigma_f / n_obs
        cov_shanken = (1.0 + c) * (cov_fm - bordered) + bordered
        se_shanken = np.sqrt(np.clip(np.diag(cov_shanken), 0.0, None))
    else:
        se_shanken = se.copy()

    mean_ret = y.mean(axis=0)
    fitted = x @ gamma
    ss_res = float(((mean_ret - fitted) ** 2).sum())
    ss_tot = float(((mean_ret - mean_ret.mean()) ** 2).sum())

    return FamaMacBethResult(
        gamma=pd.Series(gamma, index=names, name="gamma"),
        se=pd.Series(se, index=names, name="se"),
        se_shanken=pd.Series(se_shanken, index=names, name="se_shanken"),
        gamma_t=pd.DataFrame(gamma_t, index=excess_returns.index, columns=names),
        r2_cross_section=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    )
