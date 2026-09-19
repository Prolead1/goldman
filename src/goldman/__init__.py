"""FE 5108 Midterm Project — portfolio analytics package.

Public surface intentionally small: configuration, data loaders, portfolio
math, and the exhibit builders.
"""

from __future__ import annotations

from . import config, data, portfolio
from .data import load_ff3, load_returns, monthly_rf
from .portfolio import (
    equal_weights,
    excess_means,
    subsample_tangency,
    tangency_weights,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "config",
    "data",
    "portfolio",
    "load_ff3",
    "load_returns",
    "monthly_rf",
    "equal_weights",
    "excess_means",
    "subsample_tangency",
    "tangency_weights",
]