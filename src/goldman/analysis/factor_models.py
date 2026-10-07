"""Stage 3 — multi-factor regressions and the before/after alpha comparison.

Extends the Stage-2 security characteristic regression to ``K`` factors,

    ``R_i,t - R_f,t = alpha_i + sum_k beta_ik F_k,t + eps_i,t``,

estimates it for any named set of factor panels (the market alone, the market
plus self-built size and momentum, the Fama–French three factors, …), and
collects the resulting alphas so Stage 2 and Stage 3 can be compared directly.
The joint test that all alphas are zero generalises the GRS statistic in
:mod:`goldman.analysis.capm` to several factors.

All estimation uses one common sample across the models — the months where
*every* factor is available — so a smaller alpha after adding a factor cannot be
an artefact of estimating on a shorter window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .. import config
from ..data import risk_free_series
from ..viz import plot_alpha_comparison
from .capm import excess_returns, grs_test

__all__ = [
    "FactorModelResult",
    "factor_regression",
    "alpha_comparison",
    "model_summary",
    "run_factor_analysis",
]


@dataclass
class FactorModelResult:
    """Exhibits and diagnostics produced by :func:`run_factor_analysis`."""

    universe: str
    models: tuple[str, ...]
    tables: dict[str, pd.DataFrame]
    alphas: pd.DataFrame
    summary: pd.DataFrame
    grs: dict[str, dict[str, float]]
    sample: pd.DatetimeIndex
    output_paths: dict[str, Path]


def _as_datetime_index(factors: pd.DataFrame) -> pd.DataFrame:
    """Return factors on a month-start ``DatetimeIndex`` (FF files use periods)."""
    if isinstance(factors.index, pd.PeriodIndex):
        factors = factors.copy()
        factors.index = factors.index.to_timestamp()
    factors.index.name = "Date"
    return factors


def factor_regression(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    rf: float | pd.Series = 0.0,
) -> pd.DataFrame:
    """One row per asset: multi-factor ``alpha``, betas, t-statistics and ``R^2``.

    OLS is closed form and vectorised across assets.  Rows where any factor is
    missing are dropped, and ``n`` reports the effective sample per asset.
    Betas and t-statistics are named ``beta_<factor>`` / ``t_<factor>``.
    """
    excess = excess_returns(returns, rf)
    aligned = _as_datetime_index(factors).reindex(excess.index)
    valid = aligned.notna().all(axis=1)
    if not valid.any():
        raise ValueError("No overlapping factor/return observations.")
    y = excess.loc[valid].to_numpy(dtype=float)
    f = aligned.loc[valid].to_numpy(dtype=float)
    t, n_assets = y.shape
    k = f.shape[1]
    if t - n_assets - k <= 0:
        raise ValueError("Not enough observations for the factor regression.")

    design = np.column_stack([np.ones(t), f])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)  # (k + 1) x n_assets
    resid = y - design @ coefs

    dof = t - k - 1
    sigma2 = (resid**2).sum(axis=0) / dof
    xtx_inv = np.linalg.inv(design.T @ design)
    se = np.sqrt(sigma2[None, :] * np.diag(xtx_inv)[:, None])  # (k + 1) x n_assets
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = coefs / se

    sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r_squared = 1.0 - (resid**2).sum(axis=0) / sst

    data: dict[str, np.ndarray] = {
        "alpha": coefs[0],
        "t_alpha": t_stats[0],
        "r_squared": r_squared,
        "n": np.full(n_assets, t),
    }
    for position, name in enumerate(factors.columns, start=1):
        data[f"beta_{name}"] = coefs[position]
        data[f"t_{name}"] = t_stats[position]

    table = pd.DataFrame(data, index=returns.columns)
    table["p_alpha"] = 2.0 * stats.t.sf(np.abs(table["t_alpha"].to_numpy()), dof)
    table.index.name = "asset"
    return table


def alpha_comparison(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Alphas side by side, one column per model (assets in the rows)."""
    return pd.DataFrame({name: table["alpha"] for name, table in tables.items()})


def model_summary(
    tables: Mapping[str, pd.DataFrame],
    grs: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """One row per model: alpha magnitude, significance count, fit and GRS."""
    rows = {}
    for name, table in tables.items():
        test = grs.get(name, {})
        rows[name] = {
            "assets": len(table),
            "mean |alpha|": table["alpha"].abs().mean(),
            "max |alpha|": table["alpha"].abs().max(),
            "alphas p<0.05": int((table["p_alpha"] < 0.05).sum()),
            "mean R2": table["r_squared"].mean(),
            "GRS F": test.get("f_stat", np.nan),
            "GRS p": test.get("p_value", np.nan),
        }
    return pd.DataFrame(rows).T


def _slug(name: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in name)
    return cleaned.strip("_").lower()


def _common_sample(excess: pd.DataFrame, models: Mapping[str, pd.DataFrame]):
    """Index where every model's factors are all observed."""
    if not models:
        raise ValueError("At least one factor model is required.")
    valid = pd.Series(True, index=excess.index)
    for factors in models.values():
        aligned = _as_datetime_index(factors).reindex(excess.index)
        valid &= aligned.notna().all(axis=1)
    return excess.index[valid.to_numpy()]


def run_factor_analysis(
    returns: pd.DataFrame,
    *,
    universe: str,
    models: Mapping[str, pd.DataFrame],
    rf: float | pd.Series | None = None,
    output_dir: str | Path | None = None,
    write_tables: bool = True,
) -> FactorModelResult:
    """Estimate every factor model on one common sample and write the exhibits.

    ``models`` maps a display name (e.g. ``"CAPM"``, ``"Size + momentum"``,
    ``"FF3"``) to a date-by-asset factor panel; the market factor should be the
    first column so betas read naturally.  The common sample keeps the
    comparison honest, and the GRS test uses the same window as the alphas.
    """
    out_dir = config.ensure_outputs_dir(Path(output_dir) if output_dir else None)
    if rf is None:
        rf = risk_free_series(universe, returns.index)
    excess = excess_returns(returns, rf)
    sample = _common_sample(excess, models)
    if len(sample) == 0:
        raise ValueError("The factor models share no common observations.")
    returns_used = returns.loc[sample]

    tables: dict[str, pd.DataFrame] = {}
    grs: dict[str, dict[str, float]] = {}
    for name, factors in models.items():
        aligned = _as_datetime_index(factors).reindex(sample)
        tables[name] = factor_regression(returns_used, aligned, rf=rf)
        grs[name] = grs_test(excess.loc[sample], aligned)

    alphas = alpha_comparison(tables)
    summary = model_summary(tables, grs).round(4)

    output_paths: dict[str, Path] = {}
    if write_tables:
        for name, table in tables.items():
            path = out_dir / f"factors_{universe}_{_slug(name)}_scl.csv"
            table.to_csv(path)
            output_paths[f"{name} scl"] = path
        alpha_path = out_dir / f"factors_{universe}_alphas.csv"
        summary_path = out_dir / f"factors_{universe}_summary.csv"
        alphas.to_csv(alpha_path)
        summary.to_csv(summary_path)
        output_paths["alphas"] = alpha_path
        output_paths["summary"] = summary_path

    chart_path = plot_alpha_comparison(
        alphas,
        summary,
        out_dir / f"factors_{universe}_alpha_comparison.png",
        title=f"Alphas before and after factors ({universe.upper()})",
    )
    output_paths["chart"] = chart_path

    return FactorModelResult(
        universe=universe,
        models=tuple(models),
        tables=tables,
        alphas=alphas,
        summary=summary,
        grs=grs,
        sample=sample,
        output_paths=output_paths,
    )
