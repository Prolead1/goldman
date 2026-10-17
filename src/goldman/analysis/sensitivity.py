"""Opt-in regression sensitivity checks using caller-supplied branch inputs.

Compare HAC lags or asset subsets while holding model definitions, risk-free
rates and dates fixed. Functions return tables in memory; nothing is loaded,
saved or connected to the report. Supply main's prepared factor panels directly,
including a one-column panel for CAPM. No universe or 21-stock list is hardcoded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from .robust_inference import _common_inputs, _validate_lags, robust_factor_regression

__all__ = ["SensitivityResult", "hac_lag_sensitivity", "asset_subset_sensitivity"]


@dataclass
class SensitivityResult:
    """Detailed coefficient tables, alpha summaries, and the exact common sample.

    Lag results have ``(model, lag, asset)`` row indices; subset results use
    ``(subset, model, asset)``. Summary indices omit the asset level. All alpha
    magnitudes remain monthly decimals. Counts are individual significance
    screens, not joint tests or evidence that GRS assumptions have been relaxed.
    """

    tables: pd.DataFrame
    summary: pd.DataFrame
    sample: pd.DatetimeIndex


def _summary(table: pd.DataFrame) -> dict:
    return {
        "assets": len(table),
        "n": int(table.n.iloc[0]),
        "mean_abs_alpha": table.alpha.abs().mean(),
        "max_abs_alpha": table.alpha.abs().max(),
        "mean_r_squared": table.r_squared.mean(),
        "significant_alpha": int(table.significant_alpha.sum()),
        "significant_alpha_hac": int(table.significant_alpha_hac.sum()),
        "changed_alpha_significance": int(
            (table.significant_alpha != table.significant_alpha_hac).sum()
        ),
    }


def _result(tables: dict, levels: list[str], sample: pd.DatetimeIndex) -> SensitivityResult:
    summary = pd.DataFrame.from_dict(
        {key: _summary(table) for key, table in tables.items()}, orient="index"
    )
    summary.index = pd.MultiIndex.from_tuples(summary.index, names=levels)
    # pandas compares attrs on concatenation; Index-valued attrs are ambiguous.
    # The exact common calendar is retained once on SensitivityResult instead.
    cleaned = {}
    for key, table in tables.items():
        cleaned[key] = table.copy()
        cleaned[key].attrs = {}
    combined = pd.concat(cleaned, names=[*levels, "asset"])
    combined.attrs.update(
        {
            name: value
            for name, value in next(iter(tables.values())).attrs.items()
            if name != "sample"
        }
    )
    return SensitivityResult(combined, summary, sample.copy())


def hac_lag_sensitivity(
    returns: pd.DataFrame,
    models: Mapping[str, pd.DataFrame],
    rf: float | pd.Series = 0.0,
    *,
    lags: Sequence[int] = (0, 3, 6),
    significance: float = 0.05,
) -> SensitivityResult:
    """Compare classical and HAC inference over lag choices on identical dates.

    All models use the intersection of available factor months, including any
    momentum warm-up. Missing interior months raise rather than collapsing
    calendar gaps into adjacent HAC observations. Lag changes must leave OLS
    coefficients unchanged. No lag is automatically selected as 'best'.
    """
    returns, models, rf = _common_inputs(returns, models, rf)
    lags = tuple(lags)
    if not lags:
        raise ValueError("Provide at least one lag.")
    for lag in lags:
        _validate_lags(lag, len(returns))
    if len(set(lags)) != len(lags):
        raise ValueError("Lag choices must be unique.")
    tables = {
        (name, lag): robust_factor_regression(
            returns, factors, rf, lags=lag, significance=significance
        )
        for name, factors in models.items()
        for lag in lags
    }
    return _result(tables, ["model", "lag"], returns.index)


def asset_subset_sensitivity(
    returns: pd.DataFrame,
    models: Mapping[str, pd.DataFrame],
    subsets: Mapping[str, Sequence[str]],
    rf: float | pd.Series = 0.0,
    *,
    lags: int = 3,
    significance: float = 0.05,
) -> SensitivityResult:
    """Compare selected asset groups against the full supplied return panel.

    The baseline is named ``full``. Pass e.g. ``{"stocks_only": stock_names}``.
    Existing market proxies and factors are held fixed: individual coefficients
    for an asset shared by two groups are therefore identical. This isolates
    changes in the assets being summarised, not changes to portfolio weights,
    factor construction, GRS tests or out-of-sample performance. Such broader
    universe comparisons require a separate, explicitly requested analysis.
    """
    if not subsets or "full" in subsets:
        raise ValueError("Provide subsets; 'full' is reserved for the baseline.")
    groups = {"full": list(returns.columns)}
    for name, assets in subsets.items():
        if isinstance(assets, str):
            raise ValueError("Each subset must be a sequence of asset names, not a string.")
        assets = list(assets)
        if not assets or len(set(assets)) != len(assets):
            raise ValueError(f"Subset {name!r} needs nonempty, unique assets.")
        missing = set(assets) - set(returns.columns)
        if missing:
            raise ValueError(f"Unknown assets in {name!r}: {sorted(missing)}.")
        groups[name] = assets
    returns, models, rf = _common_inputs(returns, models, rf)
    # Fit once per model: dropping other response columns cannot change an
    # individual time-series regression with the same factors and calendar.
    fitted = {
        name: robust_factor_regression(returns, factors, rf, lags=lags, significance=significance)
        for name, factors in models.items()
    }
    tables = {
        (group, model): table.loc[assets].copy()
        for group, assets in groups.items()
        for model, table in fitted.items()
    }
    return _result(tables, ["subset", "model"], returns.index)
