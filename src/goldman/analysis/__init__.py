"""Analysis: portfolio construction and the report exhibits.

Currently the mean–variance portfolio choice analysis; further analyses (the
CAPM/SML, factor models, the decision memo) slot in here beside it.
"""

from __future__ import annotations

from . import portfolio, portfolio_choice
from .portfolio import (
    equal_weights,
    excess_means,
    normalize_weights,
    subsample_tangency,
    tangency_weights,
    weight_correlation,
)
from .portfolio_choice import (
    PortfolioAnalysisResult,
    plot_weights,
    run_portfolio_analysis,
    summary_statistics,
    tangency_vs_comparison,
)

__all__ = [
    "portfolio",
    "portfolio_choice",
    "PortfolioAnalysisResult",
    "run_portfolio_analysis",
    "summary_statistics",
    "tangency_vs_comparison",
    "plot_weights",
    "equal_weights",
    "excess_means",
    "normalize_weights",
    "subsample_tangency",
    "tangency_weights",
    "weight_correlation",
]