"""Alternative self-financing factors: low volatility and one-month reversal.

Signals use information available strictly before the holding month. The
same universe's returns form equal-weight long/short tercile portfolios.
"""

from __future__ import annotations

import pandas as pd

from .factor_construction import long_short_spread


def low_volatility_factor(returns: pd.DataFrame, window: int = 12) -> pd.Series:
    """Long low trailing-volatility names, short high-volatility names.

    Trailing volatility uses t-window ... t-1, excluding month t.
    """
    if window < 3:
        raise ValueError("window must be at least 3")
    signal = returns.shift(1).rolling(window, min_periods=window).std()
    return long_short_spread(returns, signal, long_high=False).rename("LOWVOL")


def reversal_factor(returns: pd.DataFrame) -> pd.Series:
    """Long previous-month losers, short previous-month winners."""
    return long_short_spread(returns, returns.shift(1), long_high=False).rename("REV")


def alternative_factors(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Two lagged, equal-weighted factor spreads for either market."""
    return pd.concat(
        [low_volatility_factor(returns, window), reversal_factor(returns)], axis=1
    )
