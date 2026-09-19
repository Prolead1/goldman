"""Data preparation: the full pre-cleaning phase.

Turns the raw cached CSVs into the clean inputs every analysis consumes:
monthly return panels and the aligned factor / risk-free series.
"""

from __future__ import annotations

from .factors import load_ff3, monthly_rf, to_period_index
from .market_caps import load_value_weights, value_weights_from_market_caps
from .prices import load_prices, load_returns, prices_to_returns

__all__ = [
    "load_ff3",
    "monthly_rf",
    "to_period_index",
    "load_prices",
    "load_returns",
    "prices_to_returns",
    "load_value_weights",
    "value_weights_from_market_caps",
]