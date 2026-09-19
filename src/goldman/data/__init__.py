"""Data preparation: the full pre-cleaning phase.

Turns the raw cached CSVs into the clean inputs every analysis consumes:
monthly return panels and the aligned factor / risk-free series.
"""

from __future__ import annotations

from .factors import load_ff3, monthly_rf, to_period_index
from .market_caps import (
    build_value_weights_from_raw,
    load_value_weights,
    market_caps_from_shares,
    unadjust_close,
    value_weights_from_market_caps,
    value_weights_from_raw,
)
from .prices import load_prices, load_returns, prices_to_returns

__all__ = [
    "load_ff3",
    "monthly_rf",
    "to_period_index",
    "load_prices",
    "load_returns",
    "prices_to_returns",
    "build_value_weights_from_raw",
    "load_value_weights",
    "market_caps_from_shares",
    "unadjust_close",
    "value_weights_from_market_caps",
    "value_weights_from_raw",
]