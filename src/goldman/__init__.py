"""FE 5108 Project — portfolio analytics package.

Two subpackages carry the work: :mod:`goldman.data` (the pre-cleaning phase)
and :mod:`goldman.analysis` (portfolio construction and the exhibits), with
:mod:`goldman.viz` holding the shared plotting theme and figures.
"""

from __future__ import annotations

from . import analysis, config, data, viz
from .analysis.portfolio import (
    equal_weights,
    excess_means,
    subsample_tangency,
    tangency_weights,
)
from .data import load_ff3, load_returns, monthly_rf

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "analysis",
    "config",
    "data",
    "viz",
    "load_ff3",
    "load_returns",
    "monthly_rf",
    "equal_weights",
    "excess_means",
    "subsample_tangency",
    "tangency_weights",
]