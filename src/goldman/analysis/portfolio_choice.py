"""Mean–variance portfolio choice and estimation error.

Builds the portfolio-choice exhibits from a clean return panel:

* summary statistics of the universe
* the tangency portfolio beside the comparison (value / equal) weights
* subsample re-estimation showing how far the weights move
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; imported before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402
from .portfolio import (  # noqa: E402
    equal_weights,
    excess_means,
    subsample_tangency,
    tangency_weights,
    weight_correlation,
)

__all__ = [
    "PortfolioAnalysisResult",
    "summary_statistics",
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


def tangency_vs_comparison(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    comparison: pd.Series | None = None,
) -> pd.DataFrame:
    """Tangency weights beside the comparison weights.

    ``comparison`` defaults to the naive 1/N portfolio.  Pass value weights to
    match the handout's requirement.
    """
    means = excess_means(returns, rf)
    tangency = tangency_weights(means, returns.cov())
    if comparison is None:
        comparison = equal_weights(returns.columns)
    comparison = comparison.reindex(returns.columns)
    return pd.DataFrame({"tangency": tangency, "comparison": comparison})


def plot_weights(
    weights: pd.DataFrame,
    output_path: str | Path,
    comparison_label: str = "comparison",
    title: str = "Estimated tangency weights vs comparison weights",
) -> Path:
    """Plot the two weight vectors as one grouped bar chart."""
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    positions = np.arange(len(weights))
    ax.bar(positions - 0.2, weights["tangency"], 0.4, label=r"tangency  $\hat\Sigma^{-1}\hat\mu_e$")
    ax.bar(positions + 0.2, weights["comparison"], 0.4, label=comparison_label)
    ax.set_xticks(positions)
    ax.set_xticklabels(weights.index, rotation=90, fontsize=7)
    ax.axhline(0, lw=0.8, color="k")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_portfolio_analysis(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    comparison: pd.Series | None = None,
    output_dir: str | Path | None = None,
) -> PortfolioAnalysisResult:
    """Run the portfolio-choice exhibits and write them to ``output_dir``."""
    out_dir = config.ensure_outputs_dir(Path(output_dir) if output_dir else None)

    summary = summary_statistics(returns, rf)
    summary_path = out_dir / "summary_stats.csv"
    summary.to_csv(summary_path)

    weights = tangency_vs_comparison(returns, rf, comparison)
    plot_path = plot_weights(weights, out_dir / "tangency_weights.png")

    drift = subsample_tangency(returns, rf).round(3)
    drift_path = out_dir / "subsample_weights.csv"
    drift.to_csv(drift_path)

    corr = weight_correlation(drift, "first half", "second half")
    most_extreme = float(np.abs(drift["full sample"]).max())

    return PortfolioAnalysisResult(
        summary=summary,
        weights=weights,
        drift=drift,
        weight_correlation=corr,
        most_extreme_weight=most_extreme,
        output_paths={"summary": summary_path, "weights": plot_path, "subsample": drift_path},
    )