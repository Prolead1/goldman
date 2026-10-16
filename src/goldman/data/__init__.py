"""Data preparation: the full pre-cleaning phase.

Turns the raw cached CSVs into the clean inputs every analysis consumes:
monthly return panels and the aligned factor / risk-free series.
"""

from __future__ import annotations

from .cleaning import build_sg_clean, load_corporate_actions, total_return_index
from .factors import load_ff3, monthly_rf, to_period_index
from .market_caps import (
    build_value_weights_from_raw,
    load_value_weights,
    market_caps_from_shares,
    unadjust_close,
    value_weights_from_market_caps,
    value_weights_from_raw,
)
from .markets import market_excess_returns
from .prices import load_prices, load_returns, prices_to_returns
from .rates import load_rf_file, monthly_rf_from_sora_export, risk_free_series

__all__ = [
    "build_sg_clean",
    "load_corporate_actions",
    "total_return_index",
    "load_ff3",
    "monthly_rf",
    "to_period_index",
    "market_excess_returns",
    "load_rf_file",
    "monthly_rf_from_sora_export",
    "risk_free_series",
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