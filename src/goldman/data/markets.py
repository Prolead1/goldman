"""Market proxies for the CAPM and factor stages.

The market excess return ``R_M - R_f`` can be built two ways, and the choice is
central to Roll's critique:

* ``"value_weighted"`` — the cap-weighted composite of the *same* universe,
  ``sum_i w_{i,t} R_{i,t} - R_f``.  Uses the point-in-time value weights and so
  works for both the US and Singapore universes.
* ``"ff_market"`` — the Fama–French ``Mkt-RF`` factor, an external US market.

Keeping both behind one function is what lets Stage 4 re-run the CAPM with a
different proxy and show how far betas and alphas move.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from .factors import load_ff3
from .market_caps import load_value_weights
from .rates import risk_free_series

__all__ = [
    "market_excess_returns",
]

MARKET_PROXIES = ("value_weighted", "ff_market")


def market_excess_returns(
    returns: pd.DataFrame,
    *,
    universe: str | None = None,
    method: str | None = None,
    value_weights: pd.DataFrame | None = None,
    ff3: pd.DataFrame | None = None,
    rf: float | pd.Series | None = None,
) -> pd.Series:
    """Return the monthly market excess return aligned to ``returns``.

    Parameters
    ----------
    returns:
        Clean monthly simple-return panel (dates by asset).
    universe:
        Universe name; supplies the default ``method``, the value weights and
        the risk-free rate.
    method:
        ``"value_weighted"`` or ``"ff_market"``; defaults to the universe's
        registered ``market_proxy`` (or ``"value_weighted"`` for a bare panel).
    value_weights:
        Date-by-asset weight panel.  Loaded from the universe when omitted.
    ff3:
        Fama–French frame used by ``"ff_market"``; loaded on demand.
    rf:
        Risk-free scalar or series used by ``"value_weighted"``; resolved from
        the universe when omitted.
    """
    if method is None:
        method = config.get_universe(universe).market_proxy if universe else "value_weighted"

    if method == "ff_market":
        if ff3 is None:
            ff3 = load_ff3()
        periods = pd.PeriodIndex(pd.to_datetime(returns.index), freq="M")
        series = ff3["Mkt-RF"].reindex(periods)
        series.index = returns.index
        if series.isna().any():
            raise ValueError("Fama–French Mkt-RF does not cover the whole return calendar.")
        return series.rename("market_excess")

    if method == "value_weighted":
        if value_weights is None:
            if universe is None:
                raise ValueError("value_weighted needs `universe` or explicit `value_weights`.")
            value_weights = load_value_weights(
                universe=universe, assets=returns.columns, dates=returns.index
            )
        weights = value_weights.reindex(index=returns.index, columns=returns.columns)
        gross = (weights * returns).sum(axis=1)
        if rf is None:
            rf = (
                risk_free_series(universe, returns.index)
                if universe is not None
                else config.RF_MONTHLY
            )
        return (gross - rf).rename("market_excess")

    raise ValueError(f"Unknown market proxy {method!r}; choose one of: {', '.join(MARKET_PROXIES)}")
