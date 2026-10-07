"""Stage 2 — the CAPM and the empirical security market line.

Estimate the security characteristic regression

    ``R_i,t - R_f,t = alpha_i + beta_i (R_M,t - R_f,t) + eps_i,t``

for every asset, draw the empirical SML (mean excess return against ``beta``)
beside the theoretical line through the origin, and jointly test that all alphas
are zero (GRS).

Everything here is universe-agnostic: pass a return panel and a market
excess-return series, or a universe name and let
:func:`goldman.data.markets.market_excess_returns` build the proxy.  Swapping the
proxy is a one-argument change — the mechanism Roll's critique turns on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .. import config
from ..data import load_ff3, risk_free_series
from ..data.markets import market_excess_returns
from ..viz import plot_sml

__all__ = [
    "SMLFit",
    "CapmResult",
    "excess_returns",
    "security_characteristic_line",
    "empirical_sml",
    "grs_test",
    "plot_sml",
    "run_capm_analysis",
]


@dataclass(frozen=True)
class SMLFit:
    """Cross-sectional fit of average excess return on estimated beta."""

    slope: float
    intercept: float
    r_squared: float
    theoretical_slope: float
    n: int


@dataclass
class CapmResult:
    """Exhibits and diagnostics produced by :func:`run_capm_analysis`."""

    universe: str
    method: str
    table: pd.DataFrame
    fit: SMLFit
    grs: dict[str, float]
    output_paths: dict[str, Path]


def excess_returns(returns: pd.DataFrame, rf: float | pd.Series) -> pd.DataFrame:
    """Subtract the risk-free rate from each asset's monthly return."""
    if isinstance(rf, pd.Series):
        return returns.sub(rf.reindex(returns.index), axis=0)
    return returns - rf


def security_characteristic_line(
    returns: pd.DataFrame,
    market_excess: pd.Series,
    rf: float | pd.Series = 0.0,
) -> pd.DataFrame:
    """One row per asset: ``alpha``, ``beta``, t-statistics, ``R^2`` and ``n``.

    OLS is closed form.  Standard errors use the usual ``(X'X)^{-1}`` formulas
    for a regression with an intercept, so ``t_alpha`` and ``t_beta`` are the
    textbook CAPM diagnostics.
    """
    x = market_excess.reindex(returns.index).astype(float)
    if x.isna().any():
        raise ValueError("market_excess does not cover the whole return calendar.")
    xv = x.to_numpy()
    xbar = xv.mean()
    x_ss = float(((xv - xbar) ** 2).sum())
    n = len(xv)
    dof = n - 2

    excess = excess_returns(returns, rf)
    rows = []
    for asset in returns.columns:
        y = excess[asset].astype(float).to_numpy()
        ybar = y.mean()
        beta = float(((xv - xbar) * (y - ybar)).sum() / x_ss)
        alpha = float(ybar - beta * xbar)
        resid = y - alpha - beta * xv
        s2 = float((resid**2).sum() / dof)
        se_beta = float(np.sqrt(s2 / x_ss))
        se_alpha = float(np.sqrt(s2 * (1.0 / n + xbar**2 / x_ss)))
        sst = float(((y - ybar) ** 2).sum())
        r_squared = 1.0 - float((resid**2).sum()) / sst if sst > 0 else np.nan
        rows.append(
            {
                "mean_excess": ybar,
                "alpha": alpha,
                "beta": beta,
                "se_alpha": se_alpha,
                "se_beta": se_beta,
                "t_alpha": alpha / se_alpha,
                "t_beta": beta / se_beta,
                "r_squared": r_squared,
                "n": n,
            }
        )

    table = pd.DataFrame(rows, index=returns.columns)
    table["p_alpha"] = 2.0 * stats.t.sf(np.abs(table["t_alpha"].to_numpy()), dof)
    table["p_beta"] = 2.0 * stats.t.sf(np.abs(table["t_beta"].to_numpy()), dof)
    table.index.name = "asset"
    return table


def empirical_sml(table: pd.DataFrame, market_excess: pd.Series) -> SMLFit:
    """Fit the empirical SML and record the theoretical slope.

    The empirical line comes from regressing each asset's average excess return
    on its estimated beta.  The theoretical line is the CAPM's prediction: it
    passes through the origin with slope equal to the average market excess
    return.
    """
    beta = table["beta"].to_numpy()
    mean_excess = table["mean_excess"].to_numpy()
    slope, intercept = np.polyfit(beta, mean_excess, 1)
    fitted = intercept + slope * beta
    ss_res = float(((mean_excess - fitted) ** 2).sum())
    ss_tot = float(((mean_excess - mean_excess.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return SMLFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        theoretical_slope=float(market_excess.mean()),
        n=len(beta),
    )


def grs_test(
    excess: pd.DataFrame,
    factors: pd.Series | pd.DataFrame,
) -> dict[str, float]:
    """Gibbons–Ross–Shanken test that all alphas are jointly zero.

    Implements the classic statistic

        ``F = (T-N-K)/N * (1 + mu' Omega^{-1} mu)^{-1} * alpha' Sigma^{-1} alpha``

    for ``K`` factors (``K = 1`` for the CAPM, ``K >= 2`` for a factor model),
    and returns the F statistic, its p-value and the degrees of freedom.  Pass a
    ``Series`` for a single factor or a ``DataFrame`` for several; the factor
    columns must be in the same order as the regression design.
    """
    factors = factors.reindex(excess.index).astype(float)
    missing = factors.isna().any(axis=1) if isinstance(factors, pd.DataFrame) else factors.isna()
    if missing.any():
        raise ValueError("factors do not cover the whole return calendar.")
    y = excess.to_numpy(dtype=float)
    f = factors.to_numpy(dtype=float)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    t, n_assets = y.shape
    k = f.shape[1]
    if t - n_assets - k <= 0:
        raise ValueError("Not enough observations for the GRS test (need T > N + K).")

    design = np.column_stack([np.ones(t), f])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha = coefs[0]  # (N,)
    resid = y - design @ coefs

    dof = t - k - 1
    sigma = resid.T @ resid / dof  # (N, N) residual covariance
    mu = f.mean(axis=0)  # (K,)
    demeaned = f - mu
    omega = demeaned.T @ demeaned / dof  # (K, K) factor covariance

    alpha = alpha.reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    sharpe_term = 1.0 + (mu.T @ np.linalg.inv(omega) @ mu).item()
    quad = (alpha.T @ np.linalg.inv(sigma) @ alpha).item()
    f_stat = float((t - n_assets - k) / n_assets * quad / sharpe_term)
    df1 = n_assets
    df2 = t - n_assets - k
    return {
        "f_stat": f_stat,
        "p_value": float(stats.f.sf(f_stat, df1, df2)),
        "df1": float(df1),
        "df2": float(df2),
        "n_assets": float(n_assets),
        "n_obs": float(t),
    }


def run_capm_analysis(
    returns: pd.DataFrame,
    *,
    universe: str,
    method: str | None = None,
    value_weights: pd.DataFrame | None = None,
    ff3: pd.DataFrame | None = None,
    rf: float | pd.Series | None = None,
    output_dir: str | Path | None = None,
) -> CapmResult:
    """Run every Stage-2 exhibit for one universe and write them to disk."""
    spec = config.get_universe(universe)
    method = method or spec.market_proxy
    out_dir = config.ensure_outputs_dir(Path(output_dir) if output_dir else None)

    if rf is None:
        rf = risk_free_series(universe, returns.index, ff3=ff3)

    # The Fama–French file is published with a lag, so a factor proxy may not
    # cover the last month of the return panel.  Estimate on the common sample
    # (stated once) rather than fail or silently fabricate a factor value.
    if method == "ff_market":
        if ff3 is None:
            ff3 = load_ff3()
        covered = pd.PeriodIndex(returns.index, freq="M").isin(ff3.index)
        if not covered.all():
            warnings.warn(
                f"{universe}: {int((~covered).sum())} month(s) of returns are not "
                "covered by the factor file; estimating on the common sample.",
                stacklevel=2,
            )
            returns = returns.loc[covered]
            if isinstance(rf, pd.Series):
                rf = rf.loc[returns.index]

    market = market_excess_returns(
        returns,
        universe=universe,
        method=method,
        value_weights=value_weights,
        ff3=ff3,
        rf=rf,
    )

    table = security_characteristic_line(returns, market, rf=rf)
    fit = empirical_sml(table, market)
    grs = grs_test(excess_returns(returns, rf), market)

    prefix = f"capm_{universe}_{method}"
    scl_path = out_dir / f"{prefix}_scl.csv"
    table.to_csv(scl_path)
    plot_path = plot_sml(
        table,
        fit,
        out_dir / f"{prefix}_sml.png",
        title=f"{universe.upper()} empirical SML ({method.replace('_', ' ')} market proxy)",
    )
    summary_path = out_dir / f"{prefix}_summary.txt"
    n_sig = int((table["p_alpha"] < 0.05).sum())
    summary_path.write_text(
        "\n".join(
            [
                f"universe: {universe} ({spec.currency})",
                f"market proxy: {method}",
                f"assets: {fit.n}  months: {int(grs['n_obs'])}",
                f"fitted SML: slope={fit.slope:.4f} "
                f"intercept={fit.intercept:.4f} R2={fit.r_squared:.4f}",
                f"theoretical slope (mean market excess): {fit.theoretical_slope:.4f}",
                f"alphas significant at 5%: {n_sig} of {fit.n}",
                f"GRS F({int(grs['df1'])},{int(grs['df2'])})="
                f"{grs['f_stat']:.3f}  p={grs['p_value']:.4f}",
                "",
            ]
        )
    )

    return CapmResult(
        universe=universe,
        method=method,
        table=table,
        fit=fit,
        grs=grs,
        output_paths={"scl": scl_path, "sml": plot_path, "summary": summary_path},
    )
