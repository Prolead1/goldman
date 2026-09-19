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
from .portfolio_choice import run_portfolio_analysis

__all__ = [
    "portfolio",
    "portfolio_choice",
    "run_portfolio_analysis",
    "equal_weights",
    "excess_means",
    "normalize_weights",
    "subsample_tangency",
    "tangency_weights",
    "weight_correlation",
]