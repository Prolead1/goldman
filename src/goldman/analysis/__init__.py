"""Analysis: portfolio construction and the report exhibits.

Currently the mean–variance portfolio choice analysis; further analyses (the
CAPM/SML, factor models, the decision memo) slot in here beside it.
"""

from __future__ import annotations

from . import capm, portfolio, portfolio_choice
from .capm import (
    CapmResult,
    SMLFit,
    empirical_sml,
    excess_returns,
    grs_test,
    run_capm_analysis,
    security_characteristic_line,
)
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
    "capm",
    "portfolio",
    "portfolio_choice",
    "CapmResult",
    "SMLFit",
    "empirical_sml",
    "excess_returns",
    "grs_test",
    "run_capm_analysis",
    "security_characteristic_line",
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