"""Data preparation: the full pre-cleaning phase.

Turns the raw cached CSVs into the clean inputs every analysis consumes:
monthly return panels and the aligned factor / risk-free series.
"""

from __future__ import annotations

from .factors import (
    LongHistory,
    align_excess_returns,
    load_developed,
    load_factors,
    load_ff3,
    load_ff5,
    load_long_history,
    load_momentum,
    load_test_portfolios,
    monthly_rf,
    to_period_index,
)
from .market_caps import (
    build_value_weights_from_raw,
    load_holding_value_weights,
    load_value_weights,
    market_caps_from_shares,
    unadjust_close,
    value_weights_from_market_caps,
    value_weights_from_raw,
)
from .prices import load_index_proxy_returns, load_prices, load_returns, prices_to_returns

__all__ = [
    "LongHistory",
    "align_excess_returns",
    "load_developed",
    "load_long_history",
    "load_factors",
    "load_ff3",
    "load_ff5",
    "load_momentum",
    "load_test_portfolios",
    "monthly_rf",
    "to_period_index",
    "load_index_proxy_returns",
    "load_prices",
    "load_returns",
    "prices_to_returns",
    "build_value_weights_from_raw",
    "load_holding_value_weights",
    "load_value_weights",
    "market_caps_from_shares",
    "unadjust_close",
    "value_weights_from_market_caps",
    "value_weights_from_raw",
]