"""Mean–variance portfolio choice and estimation error.

Builds the portfolio-choice exhibits from a clean return panel:

* summary statistics of the universe
* the tangency portfolio beside the comparison (value / equal) weights
* subsample re-estimation showing how far the weights move
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..viz import plot_weights
from .portfolio import (
    equal_weights,
    excess_means,
    subsample_tangency,
    tangency_weights,
    weight_correlation,
)

__all__ = [
    "PortfolioAnalysisResult",
    "summary_statistics",
    "representative_weights",
    "tangency_vs_comparison",
    "plot_weights",
    "run_portfolio_analysis",
]


@dataclass
class PortfolioAnalysisResult:
    """Paths and headline diagnostics produced by :func:`run_portfolio_analysis`."""

    summary: pd.DataFrame
    weights: pd.DataFrame
    drift: pd.DataFrame
    weight_correlation: float
    most_extreme_weight: float
    output_paths: dict[str, Path]


def summary_statistics(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
) -> pd.DataFrame:
    """Annualised mean, vol and Sharpe, plus the monthly extremes."""
    if isinstance(rf, pd.Series):
        rf_mean = rf.reindex(returns.index).mean()
    else:
        rf_mean = rf
    stats = pd.DataFrame(
        {
            "mean (ann.)": returns.mean() * config.TRADING_MONTHS,
            "vol (ann.)": returns.std() * np.sqrt(config.TRADING_MONTHS),
            "sharpe (ann.)": (returns.mean() - rf_mean)
            / returns.std()
            * np.sqrt(config.TRADING_MONTHS),
            "min month": returns.min(),
            "max month": returns.max(),
        }
    )
    return stats.round(3)


def representative_weights(
    weights: pd.Series | pd.DataFrame,
    how: str = "mean",
) -> pd.Series:
    """Collapse a date-by-asset weight panel to one vector for charting.

    A ``Series`` passes through.  A ``DataFrame`` (time-varying value weights)
    is reduced with ``mean`` (sample-average composition, the default),
    ``first`` or ``last``.
    """
    if isinstance(weights, pd.Series):
        return weights
    if how == "mean":
        return weights.mean()
    if how == "first":
        return weights.iloc[0]
    if how == "last":
        return weights.iloc[-1]
    raise ValueError("how must be one of: mean, first, last")


def tangency_vs_comparison(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    comparison: pd.Series | pd.DataFrame | None = None,
    comparison_how: str = "mean",
) -> pd.DataFrame:
    """Tangency weights beside the comparison weights.

    ``comparison`` defaults to the naive 1/N portfolio.  Pass a value-weight
    series or a date-by-asset value-weight panel (reduced via
    :func:`representative_weights`) to match the handout's requirement.
    """
    means = excess_means(returns, rf)
    tangency = tangency_weights(means, returns.cov())
    if comparison is None:
        comparison = equal_weights(returns.columns)
    comparison = representative_weights(comparison, how=comparison_how)
    comparison = comparison.reindex(returns.columns)
    return pd.DataFrame({"tangency": tangency, "comparison": comparison})


def run_portfolio_analysis(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    comparison: pd.Series | pd.DataFrame | None = None,
    comparison_label: str = "1/N",
    output_dir: str | Path | None = None,
) -> PortfolioAnalysisResult:
    """Run the portfolio-choice exhibits and write them to ``output_dir``."""
    out_dir = config.ensure_outputs_dir(Path(output_dir) if output_dir else None)

    summary = summary_statistics(returns, rf)
    summary_path = out_dir / "summary_stats.csv"
    summary.to_csv(summary_path)

    # Persist the month-by-month comparison weights when they vary over time,
    # so the point-in-time benchmark is auditable rather than a single snapshot.
    value_weights_path = None
    if isinstance(comparison, pd.DataFrame):
        value_weights_path = out_dir / "value_weights.csv"
        comparison.to_csv(value_weights_path)

    weights = tangency_vs_comparison(returns, rf, comparison)
    plot_path = plot_weights(
        weights, out_dir / "tangency_weights.png", comparison_label=comparison_label
    )

    drift = subsample_tangency(returns, rf).round(3)
    drift_path = out_dir / "subsample_weights.csv"
    drift.to_csv(drift_path)

    corr = weight_correlation(drift, "first half", "second half")
    most_extreme = float(np.abs(drift["full sample"]).max())

    output_paths = {"summary": summary_path, "weights": plot_path, "subsample": drift_path}
    if value_weights_path is not None:
        output_paths["value_weights"] = value_weights_path

    return PortfolioAnalysisResult(
        summary=summary,
        weights=weights,
        drift=drift,
        weight_correlation=corr,
        most_extreme_weight=most_extreme,
        output_paths=output_paths,
    )