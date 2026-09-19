"""Exhibit builders for the empirical analyses.

One module per analysis: mean–variance portfolio choice, the CAPM/SML,
factor models, and the decision memo.
"""

from __future__ import annotations

from . import portfolio_choice

__all__ = ["portfolio_choice"]