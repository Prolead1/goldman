"""Mean–variance portfolio choice and estimation error.

Builds the portfolio-choice exhibits from a clean return panel:

* summary statistics of the universe
* the tangency portfolio beside the comparison (value / equal) weights
* subsample re-estimation showing how far the weights move
* how a no-short constraint and shrinkage tame the weights (1c)
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
from .estimators import capm_prior_means, ledoit_wolf_covariance, shrink_means  # noqa: E402
from .portfolio import (  # noqa: E402
    _tangency_for,
    equal_weights,
    excess_returns,
    subsample_tangency,
    tangency_weights,
    tangency_weights_long_only,
    weight_correlation,
)

__all__ = [
    "PortfolioAnalysisResult",
    "summary_statistics",
    "representative_weights",
    "tangency_vs_comparison",
    "plot_weights",
    "TamingResult",
    "taming_comparison",
    "subset_tangency_check",
    "plot_taming",
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
    """Annualised mean, vol and Sharpe, plus the monthly extremes.

    Mean and volatility are of total returns; the Sharpe ratio is the mean
    excess return over the volatility of *excess* returns.
    """
    excess = excess_returns(returns, rf)
    stats = pd.DataFrame(
        {
            "mean (ann.)": returns.mean() * config.TRADING_MONTHS,
            "vol (ann.)": returns.std() * np.sqrt(config.TRADING_MONTHS),
            "sharpe (ann.)": excess.mean() / excess.std() * np.sqrt(config.TRADING_MONTHS),
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
    excess = excess_returns(returns, rf)
    tangency = tangency_weights(excess.mean(), excess.cov())
    if comparison is None:
        comparison = equal_weights(returns.columns)
    comparison = representative_weights(comparison, how=comparison_how)
    comparison = comparison.reindex(returns.columns)
    return pd.DataFrame({"tangency": tangency, "comparison": comparison})


def plot_weights(
    weights: pd.DataFrame,
    output_path: str | Path,
    comparison_label: str = "comparison",
    title: str = "Estimated tangency weights vs comparison weights",
) -> Path:
    """Plot the two weight vectors as a shared-x, two-panel chart.

    The tangency weights are typically an order of magnitude larger than the
    comparison weights (extreme long–short positions versus small positive
    index weights), so the two series get their own panel and y-scale rather
    than one grouped axis where the comparison bars vanish.
    """
    output_path = Path(output_path)
    positions = np.arange(len(weights))
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax_top.bar(
        positions,
        weights["tangency"],
        0.7,
        color="#1f77b4",
        label=r"tangency  $\hat\Sigma^{-1}\hat\mu_e$",
    )
    ax_top.axhline(0, lw=0.8, color="k")
    ax_top.set_ylabel("tangency weight")
    ax_top.legend(loc="upper right")
    ax_top.set_title(title)

    ax_bot.bar(positions, weights["comparison"], 0.7, color="#ff7f0e", label=comparison_label)
    ax_bot.axhline(0, lw=0.8, color="k")
    ax_bot.set_ylabel("comparison")
    ax_bot.legend(loc="upper right")
    ax_bot.set_xticks(positions)
    ax_bot.set_xticklabels(weights.index, rotation=90, fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ------------------------------------------------------- (1c) taming
_TAMING_METHODS = [
    "tangency (plug-in)",
    "tangency (long-only)",
    "tangency (shrinkage)",
    "tangency (shrinkage, long-only)",
]


def _taming_weights(
    excess: pd.DataFrame,
    market_weights: pd.Series,
    prior_premium: float,
) -> tuple[dict[str, pd.Series], float]:
    """The four tangency variants on one sample; also the shrinkage intensity."""
    means, cov = excess.mean(), excess.cov()
    lw_cov, _ = ledoit_wolf_covariance(excess)
    prior = capm_prior_means(lw_cov, market_weights, prior_premium)
    shrunk, phi = shrink_means(means, prior, lw_cov, len(excess))
    return {
        "tangency (plug-in)": tangency_weights(means, cov),
        "tangency (long-only)": tangency_weights_long_only(means, cov),
        "tangency (shrinkage)": tangency_weights(shrunk, lw_cov),
        "tangency (shrinkage, long-only)": tangency_weights_long_only(shrunk, lw_cov),
    }, phi


@dataclass
class TamingResult:
    """Full-sample weights of each variant and their stability diagnostics."""

    weights: pd.DataFrame
    table: pd.DataFrame


def taming_comparison(
    excess_returns: pd.DataFrame,
    market_weights: pd.Series,
    prior_premium: float,
) -> TamingResult:
    """Exhibit (1c): does a no-short constraint or shrinkage tame the weights?

    For the plug-in tangency portfolio and three tamed variants (long-only;
    Ledoit–Wolf covariance with means shrunk toward the CAPM prior; both) this
    reports the size of the positions and — the estimation-error test — how
    similar the weights estimated on the two halves of the sample are, both as
    a correlation and as the L1 distance (the turnover needed to move from one
    half's portfolio to the other's).
    ``market_weights`` is the value-weight vector the CAPM prior is built on;
    ``prior_premium`` the monthly market premium it assumes.
    """
    market_weights = market_weights.reindex(excess_returns.columns)
    market_weights = market_weights / market_weights.sum()
    half = len(excess_returns) // 2
    full, phi_full = _taming_weights(excess_returns, market_weights, prior_premium)
    first, _ = _taming_weights(excess_returns.iloc[:half], market_weights, prior_premium)
    second, _ = _taming_weights(excess_returns.iloc[half:], market_weights, prior_premium)

    mu = excess_returns.mean()
    cov = excess_returns.cov()
    rows = {}
    for name in _TAMING_METHODS:
        w = full[name]
        rows[name] = {
            "max |weight|": float(w.abs().max()),
            "gross exposure": float(w.abs().sum()),
            "short positions": int((w < -1e-8).sum()),
            "half-sample weight corr": float(first[name].corr(second[name])),
            # Turnover needed to move from the first-half to the second-half
            # portfolio: how much the estimate moves, in units of capital.
            "half-sample L1 distance": float((first[name] - second[name]).abs().sum()),
            "in-sample sharpe (ann.)": float(
                w @ mu / np.sqrt(w @ cov @ w) * np.sqrt(config.TRADING_MONTHS)
            ),
            "weight on CAPM prior": phi_full if "shrinkage" in name else 0.0,
        }
    vw = market_weights
    rows["value weights"] = {
        "max |weight|": float(vw.max()),
        "gross exposure": 1.0,
        "short positions": 0,
        "half-sample weight corr": float("nan"),
        "half-sample L1 distance": float("nan"),
        "in-sample sharpe (ann.)": float(
            vw @ mu / np.sqrt(vw @ cov @ vw) * np.sqrt(config.TRADING_MONTHS)
        ),
        "weight on CAPM prior": 1.0,
    }
    table = pd.DataFrame(rows).T
    table.index.name = "portfolio"
    weights = pd.DataFrame({**full, "value weights": vw})
    weights.index.name = "asset"
    return TamingResult(weights=weights, table=table)


def subset_tangency_check(
    excess_returns: pd.DataFrame,
    subsets: dict[str, list[str]],
) -> pd.DataFrame:
    """Is the extreme tangency portfolio caused by near-duplicate assets?

    Sector ETFs hold the stocks in the universe, so some columns of the return
    panel are close to linear combinations of others.  That makes the sample
    covariance ill-conditioned (large condition number), and its inverse turns
    small differences in estimated means into huge offsetting long–short
    positions.  For each subset this reports the condition number of the
    correlation matrix and the plug-in tangency portfolio's size and stability.
    """
    rows = {}
    half = len(excess_returns) // 2
    for name, cols in subsets.items():
        sub = excess_returns[cols]
        w = _tangency_for(sub, 0.0)
        first = _tangency_for(sub.iloc[:half], 0.0)
        second = _tangency_for(sub.iloc[half:], 0.0)
        rows[name] = {
            "assets": len(cols),
            "condition number (corr)": float(np.linalg.cond(sub.corr().to_numpy())),
            "max |weight|": float(w.abs().max()),
            "gross exposure": float(w.abs().sum()),
            "short positions": int((w < 0).sum()),
            "half-sample weight corr": float(first.corr(second)),
            "half-sample L1 distance": float((first - second).abs().sum()),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "universe"
    return out


def plot_taming(
    taming: TamingResult,
    output_path: str | Path,
    title: str = "Taming the tangency portfolio",
) -> Path:
    """Small multiples: each tangency variant (bars) against the value weights (dots).

    Every panel has its own y-scale, so the plug-in portfolio's extreme
    positions do not flatten the tamed ones.
    """
    output_path = Path(output_path)
    weights = taming.weights
    positions = np.arange(len(weights))
    fig, axes = plt.subplots(len(_TAMING_METHODS), 1, figsize=(13, 11), sharex=True)
    for ax, name in zip(axes, _TAMING_METHODS, strict=True):
        ax.bar(positions, weights[name], 0.7, color="#2a78d6", label=name)
        ax.scatter(
            positions, weights["value weights"], s=22, color="#eb6834", zorder=3,
            label="value weights",
        )
        ax.axhline(0, lw=0.8, color="k")
        row = taming.table.loc[name]
        ax.set_title(
            f"{name}:  max |w| {row['max |weight|']:.0%},  "
            f"half-sample L1 distance {row['half-sample L1 distance']:.2f},  "
            f"weight corr {row['half-sample weight corr']:.2f}",
            loc="left", fontsize=10,
        )
        ax.set_ylabel("weight")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(weights.index, rotation=90, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


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