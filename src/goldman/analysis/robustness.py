"""Taming the tangency portfolio.

Stage 1 shows that the unconstrained tangency portfolio takes absurd long and
short positions because it inverts a noisy covariance matrix and amplifies a
noisy mean. Two standard fixes are compared here:

* a **no-short constraint**, which solves for the long-only portfolio of maximum
  Sharpe instead of the algebraic tangency; and
* **shrinkage**, which pulls the covariance toward its diagonal and the means
  toward their cross-sectional average before inverting.

The module reports how far each fix reduces the extremes, how much more stable
the weights are across subsamples, and how the variants perform out of sample
when fitted on one half of the data and tested on the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .. import config
from ..viz import plot_robustness
from .portfolio import excess_means, tangency_weights

__all__ = [
    "VARIANTS",
    "LABELS",
    "RobustnessResult",
    "shrink_covariance",
    "shrink_means",
    "long_only_tangency",
    "variant_weights",
    "tangency_variants",
    "weight_diagnostics",
    "variant_stability",
    "out_of_sample_sharpe",
    "plot_robustness",
    "run_robustness_analysis",
]

#: The four portfolio constructions compared by the robustness analysis.
VARIANTS = ("unconstrained", "long_only", "shrunk", "shrunk_long_only")

LABELS = {
    "unconstrained": "unconstrained",
    "long_only": "no short",
    "shrunk": "shrunk",
    "shrunk_long_only": "shrunk, no short",
}


@dataclass
class RobustnessResult:
    """Weights, diagnostics and exhibit paths for the robustness comparison."""

    weights: pd.DataFrame
    diagnostics: pd.DataFrame
    output_paths: dict[str, Path]


def _align_rf(index: pd.Index, rf: float | pd.Series) -> float | pd.Series:
    return rf.reindex(index) if isinstance(rf, pd.Series) else rf


def shrink_covariance(cov: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Shrink a covariance matrix toward its diagonal.

    ``Sigma_s = (1 - delta) * Sigma + delta * diag(Sigma)`` keeps the variances
    and damps the correlations, which is where most of the sampling noise sits.
    ``delta`` runs from 0 (no shrinkage) to 1 (diagonal).
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be between 0 and 1.")
    values = cov.to_numpy(dtype=float)
    diagonal = np.diag(np.diag(values))
    shrunk = (1.0 - delta) * values + delta * diagonal
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


def shrink_means(means: pd.Series, delta: float) -> pd.Series:
    """Shrink expected excess returns toward their cross-sectional average."""
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be between 0 and 1.")
    return (1.0 - delta) * means + delta * means.mean()


def long_only_tangency(means: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Long-only maximum Sharpe portfolio: ``max w'mu / sqrt(w'Sigma w)``.

    Subject to ``w >= 0`` and ``sum(w) = 1``.  The unconstrained analogue is
    :func:`goldman.analysis.portfolio.tangency_weights`.
    """
    mean_vector = means.to_numpy(dtype=float)
    cov_array = np.asarray(cov, dtype=float)
    n = len(mean_vector)

    def negative_sharpe(weights: np.ndarray) -> float:
        variance = float(weights @ cov_array @ weights)
        if variance <= 0:
            return 1e9
        return -float(weights @ mean_vector) / np.sqrt(variance)

    result = minimize(
        negative_sharpe,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Long-only optimisation failed: {result.message}")
    weights = np.clip(result.x, 0.0, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=means.index, name=means.name)


def variant_weights(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    variant: str = "unconstrained",
    cov_delta: float = 0.5,
    mean_delta: float = 0.5,
) -> pd.Series:
    """Build one portfolio variant from a return panel."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose one of: {', '.join(VARIANTS)}.")

    means = excess_means(returns, _align_rf(returns.index, rf))
    cov = returns.cov()
    shrunk = variant.startswith("shrunk")
    if shrunk:
        means = shrink_means(means, mean_delta)
        cov = shrink_covariance(cov, cov_delta)

    if variant.endswith("long_only"):
        weights = long_only_tangency(means, cov)
    else:
        weights = tangency_weights(means, cov)
    return weights.rename(variant)


def tangency_variants(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    cov_delta: float = 0.5,
    mean_delta: float = 0.5,
) -> pd.DataFrame:
    """All four variants as a date-free weight matrix (assets by variant)."""
    return pd.DataFrame(
        {
            variant: variant_weights(returns, rf, variant, cov_delta, mean_delta)
            for variant in VARIANTS
        }
    )


def weight_diagnostics(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarise how extreme each weight vector is."""
    rows = {}
    for column in weights.columns:
        vector = weights[column]
        rows[column] = {
            "max weight": float(vector.max()),
            "min weight": float(vector.min()),
            "shorts": int((vector < -1e-8).sum()),
            "gross leverage": float(vector.abs().sum()),
            "concentration": float((vector**2).sum()),
        }
    return pd.DataFrame(rows).T


def variant_stability(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    cov_delta: float = 0.5,
    mean_delta: float = 0.5,
) -> pd.Series:
    """Correlation between the weights estimated on each half of the sample."""
    midpoint = len(returns) // 2
    first, second = returns.iloc[:midpoint], returns.iloc[midpoint:]
    correlations = {}
    for variant in VARIANTS:
        left = variant_weights(first, _align_rf(first.index, rf), variant, cov_delta, mean_delta)
        right = variant_weights(second, _align_rf(second.index, rf), variant, cov_delta, mean_delta)
        correlations[variant] = float(left.corr(right))
    return pd.Series(correlations, name="half sample weight correlation")


def out_of_sample_sharpe(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    cov_delta: float = 0.5,
    mean_delta: float = 0.5,
) -> pd.Series:
    """Annualised Sharpe from fitting on one half and testing on the other.

    Each variant is fitted twice, on the first half then evaluated on the second
    and the reverse; the two Sharpes are averaged.  This is the honest test of a
    construction, because no weights ever see the returns they are scored on.
    """
    midpoint = len(returns) // 2
    halves = [
        (returns.iloc[:midpoint], returns.iloc[midpoint:]),
        (returns.iloc[midpoint:], returns.iloc[:midpoint]),
    ]
    scores = {}
    for variant in VARIANTS:
        sharpes = []
        for train, test in halves:
            weights = variant_weights(
                train, _align_rf(train.index, rf), variant, cov_delta, mean_delta
            )
            portfolio = pd.Series(
                test.to_numpy(dtype=float) @ weights.to_numpy(dtype=float), index=test.index
            )
            excess = portfolio - _align_rf(test.index, rf)
            if float(excess.std()) > 0:
                score = float(excess.mean()) / float(excess.std())
                sharpes.append(score * np.sqrt(config.TRADING_MONTHS))
        scores[variant] = float(np.mean(sharpes)) if sharpes else float("nan")
    return pd.Series(scores, name="out of sample Sharpe")


def run_robustness_analysis(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    cov_delta: float = 0.5,
    mean_delta: float = 0.5,
    output_dir: str | Path | None = None,
    label: str = "us",
) -> RobustnessResult:
    """Compare the four variants and write the exhibit to ``output_dir``."""
    out_dir = config.ensure_outputs_dir(Path(output_dir) if output_dir else None)

    weights = tangency_variants(returns, rf, cov_delta, mean_delta)
    diagnostics = weight_diagnostics(weights)
    diagnostics["half sample weight correlation"] = variant_stability(
        returns, rf, cov_delta, mean_delta
    )
    diagnostics["out of sample Sharpe"] = out_of_sample_sharpe(returns, rf, cov_delta, mean_delta)
    diagnostics = diagnostics.round(3)

    weights_path = out_dir / f"robustness_{label}_weights.csv"
    diagnostics_path = out_dir / f"robustness_{label}_diagnostics.csv"
    chart_path = plot_robustness(
        weights,
        diagnostics,
        out_dir / f"robustness_{label}.png",
        title=f"Taming the tangency portfolio ({label.upper()})",
    )
    weights.to_csv(weights_path)
    diagnostics.to_csv(diagnostics_path)

    return RobustnessResult(
        weights=weights,
        diagnostics=diagnostics,
        output_paths={
            "weights": weights_path,
            "diagnostics": diagnostics_path,
            "chart": chart_path,
        },
    )
