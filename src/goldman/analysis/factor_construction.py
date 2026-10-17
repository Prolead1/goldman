"""Stage 3 — self-built long–short factor portfolios.

The Fama–French library supplies the US factors, but Stage 3 asks for factors
that also exist *outside* the US and rewards building them.  This module builds
two standard factors from the same universe the rest of the project uses, so the
identical construction runs on the US and Singapore panels:

* **Size (SMB)** — every month, rank the assets by their market capitalisation at
  the *previous* month end (the committed point-in-time value weights, so the
  sort carries no look-ahead), go long the small-cap third and short the
  large-cap third, equal-weighted.
* **Momentum (WML)** — rank by the cumulative return over months ``t-12`` to
  ``t-2`` (the conventional 12-month formation that skips the most recent month
  to avoid short-term reversal), go long the winners and short the losers.

Both legs are equal-weighted, and the spread is the difference between the long
and short leg means: ``factor_t = mean(r_long,t) - mean(r_short,t)``.  The
convention follows Fama and French's 30/40/30 breakpoints — the middle third is
left out — although any ``n_quantiles`` is allowed.

Because the factors are built from the same returns they help explain, they
absorb cross-sectional variation by construction; that is the point of the
exercise and is stated in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "long_short_spread",
    "long_short_weights",
    "turnover",
    "size_factor",
    "size_weights",
    "momentum_signal",
    "momentum_factor",
    "low_volatility_factor",
    "low_volatility_weights",
    "reversal_factor",
    "reversal_weights",
    "Tradability",
    "factor_tradability",
    "momentum_tradability",
    "size_tradability",
    "self_factor_tradability",
    "build_self_factors",
]


def long_short_spread(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    n_quantiles: int = 3,
    long_high: bool = True,
) -> pd.Series:
    """Equal-weighted long–short spread sorted on ``signal`` each month.

    For every date the assets with a non-missing signal are ranked; the top
    ``1/n_quantiles`` form one leg and the bottom ``1/n_quantiles`` the other.
    ``long_high`` puts the long leg on the highest signal (momentum); set it to
    ``False`` for a low-signal long (size).  Dates with too few assets to fill
    both legs return ``NaN``.

    Parameters
    ----------
    returns:
        Monthly simple-return panel (dates by asset).
    signal:
        Date-by-asset ranking variable known *before* each return.  Missing
        entries drop that asset from the cross-section for that month.
    n_quantiles:
        Breakpoint count; 3 gives the conventional 30/40/30 split.
    long_high:
        ``True`` longs the top signal quantile, ``False`` the bottom.
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2.")

    values: dict[object, float] = {}
    for date in returns.index:
        if date not in signal.index:
            values[date] = np.nan
            continue
        scores = signal.loc[date].dropna()
        # Only assets that actually traded this month can enter a leg.
        scores = scores[scores.index.isin(returns.columns)]
        scores = scores[returns.loc[date, scores.index].notna()]
        leg = len(scores) // n_quantiles
        if leg < 1 or len(scores) < 2 * leg:
            values[date] = np.nan
            continue
        ordered = scores.sort_values(kind="stable")
        low = ordered.index[:leg]
        high = ordered.index[-leg:]
        long_leg, short_leg = (high, low) if long_high else (low, high)
        month = returns.loc[date]
        values[date] = float(month[long_leg].mean() - month[short_leg].mean())

    result = pd.Series(values, name="spread").sort_index()
    return result


def long_short_weights(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    n_quantiles: int = 3,
    long_high: bool = True,
) -> pd.DataFrame:
    """The holdings behind :func:`long_short_spread`, one row per month.

    Each long name carries ``+1/leg`` and each short name ``-1/leg``, so
    ``(weights * returns).sum(axis=1)`` reproduces the spread.  Months the
    spread leaves as ``NaN`` are all-zero rows.  The weights are what turnover
    and trading costs are measured on.
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2.")

    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for date in returns.index:
        if date not in signal.index:
            continue
        scores = signal.loc[date].dropna()
        scores = scores[scores.index.isin(returns.columns)]
        scores = scores[returns.loc[date, scores.index].notna()]
        leg = len(scores) // n_quantiles
        if leg < 1 or len(scores) < 2 * leg:
            continue
        ordered = scores.sort_values(kind="stable")
        low, high = ordered.index[:leg], ordered.index[-leg:]
        long_leg, short_leg = (high, low) if long_high else (low, high)
        weights.loc[date, long_leg] = 1.0 / leg
        weights.loc[date, short_leg] = -1.0 / leg
    return weights


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Monthly turnover: the sum of absolute weight changes across assets.

    The first month a portfolio is formed counts its full gross position.
    Drift between rebalances is ignored, which slightly understates the trades.
    """
    held = weights.fillna(0.0)
    active = held.abs().sum(axis=1) > 0
    change = held.diff().abs().sum(axis=1)
    first = active.idxmax() if active.any() else None
    if first is not None:
        change.loc[first] = held.loc[first].abs().sum()
    return change.where(active | active.shift(1, fill_value=False)).rename("turnover")


def size_factor(
    returns: pd.DataFrame,
    value_weights: pd.DataFrame,
    *,
    n_quantiles: int = 3,
) -> pd.Series:
    """SMB: long small caps, short large caps, sorted on lagged value weights.

    ``value_weights`` are the point-in-time capitalisation weights; shifting
    them by one month means month ``t`` is sorted on information available at
    the end of ``t-1``.  A larger weight is a larger firm, so ``long_high`` is
    ``False``.
    """
    weights = value_weights.reindex(index=returns.index, columns=returns.columns)
    lagged = weights.shift(1)
    spread = long_short_spread(returns, lagged, n_quantiles=n_quantiles, long_high=False)
    return spread.rename("SMB")


def size_weights(
    returns: pd.DataFrame,
    value_weights: pd.DataFrame,
    *,
    n_quantiles: int = 3,
) -> pd.DataFrame:
    """The holdings behind :func:`size_factor`, one row per month.

    Mirrors :func:`size_factor`: the value weights are shifted one month and
    the smallest third is held long, so ``(weights * returns).sum(axis=1)``
    reproduces SMB.  The weights are what turnover and trading costs are
    measured on.
    """
    weights = value_weights.reindex(index=returns.index, columns=returns.columns)
    lagged = weights.shift(1)
    return long_short_weights(returns, lagged, n_quantiles=n_quantiles, long_high=False)


def momentum_signal(
    returns: pd.DataFrame,
    *,
    lookback: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """Cumulative formation return from ``t-lookback`` to ``t-skip`` inclusive.

    The default ``12/1`` skips the most recent month: at date ``t`` the signal
    is the compounded return over months ``t-12`` to ``t-2``, every one of them
    realised before the month whose return the factor spans.
    """
    if lookback <= skip:
        raise ValueError("lookback must exceed skip.")
    window = lookback - skip
    growth = (
        (1.0 + returns)
        .shift(skip + 1)
        .rolling(window=window, min_periods=window)
        .apply(np.prod, raw=True)
    )
    return growth - 1.0


def momentum_factor(
    returns: pd.DataFrame,
    *,
    lookback: int = 12,
    skip: int = 1,
    n_quantiles: int = 3,
) -> pd.Series:
    """WML: long past winners, short past losers (12-1 momentum by default)."""
    signal = momentum_signal(returns, lookback=lookback, skip=skip)
    spread = long_short_spread(returns, signal, n_quantiles=n_quantiles, long_high=True)
    return spread.rename("WML")


def low_volatility_factor(
    returns: pd.DataFrame,
    *,
    window: int = 12,
    n_quantiles: int = 3,
) -> pd.Series:
    """LOWVOL: long the calmest third, short the most volatile third.

    The signal is each asset's standard deviation over months ``t-window`` to
    ``t-1`` (the sort never sees the month it predicts); ``long_high=False``
    puts the long leg on the lowest volatility.
    """
    signal = returns.shift(1).rolling(window, min_periods=window).std()
    spread = long_short_spread(returns, signal, n_quantiles=n_quantiles, long_high=False)
    return spread.rename("LOWVOL")


def reversal_factor(
    returns: pd.DataFrame,
    *,
    n_quantiles: int = 3,
) -> pd.Series:
    """REV: long last month's losers, short last month's winners (1-month reversal)."""
    spread = long_short_spread(returns, returns.shift(1), n_quantiles=n_quantiles, long_high=False)
    return spread.rename("REV")


def low_volatility_weights(
    returns: pd.DataFrame,
    *,
    window: int = 12,
    n_quantiles: int = 3,
) -> pd.DataFrame:
    """The holdings behind :func:`low_volatility_factor`, one row per month."""
    signal = returns.shift(1).rolling(window, min_periods=window).std()
    return long_short_weights(returns, signal, n_quantiles=n_quantiles, long_high=False)


def reversal_weights(
    returns: pd.DataFrame,
    *,
    n_quantiles: int = 3,
) -> pd.DataFrame:
    """The holdings behind :func:`reversal_factor`, one row per month."""
    return long_short_weights(
        returns, returns.shift(1), n_quantiles=n_quantiles, long_high=False
    )


def build_self_factors(
    returns: pd.DataFrame,
    *,
    value_weights: pd.DataFrame | None = None,
    universe: str | None = None,
    n_quantiles: int = 3,
    lookback: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """Build the self-made size and momentum factors for one universe.

    ``value_weights`` is the committed date-by-asset weight panel; pass it
    directly or give ``universe`` to load it.  The frame has columns ``SMB`` and
    ``WML`` aligned to ``returns`` (``WML`` begins after the momentum warm-up).
    """
    if value_weights is None:
        if universe is None:
            raise ValueError("Provide either `value_weights` or `universe`.")
        from ..data import load_value_weights

        value_weights = load_value_weights(
            universe=universe, assets=returns.columns, dates=returns.index
        )

    factors = pd.DataFrame(
        {
            "SMB": size_factor(returns, value_weights, n_quantiles=n_quantiles),
            "WML": momentum_factor(
                returns, lookback=lookback, skip=skip, n_quantiles=n_quantiles
            ),
            "LOWVOL": low_volatility_factor(returns, n_quantiles=n_quantiles),
            "REV": reversal_factor(returns, n_quantiles=n_quantiles),
        }
    )
    factors.index.name = returns.index.name
    return factors


@dataclass(frozen=True)
class Tradability:
    """Turnover, trading-cost sensitivity and drawdown of a momentum spread."""

    factor: pd.Series
    turnover: pd.Series
    drawdown: pd.Series
    break_even_bps: float

    def cost_curve(self, costs_bps=(0, 10, 25, 50, 100)) -> pd.Series:
        """Annualised mean return net of each per-unit-turnover cost.

        A cost of ``c`` basis points is charged on turnover each month, so the
        net mean is linear in ``c`` and reaches zero at :attr:`break_even_bps`.
        """
        costs = np.asarray(costs_bps, dtype=float)
        annual = self.factor.mean() * 12 - self.turnover.mean() * costs / 1e4 * 12
        return pd.Series(annual, index=costs, name="net_mean_annual")


def factor_tradability(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    name: str = "factor",
) -> Tradability:
    """Turnover, cost sensitivity and drawdown of a long-short factor.

    ``weights`` are the holdings behind the factor (see
    :func:`long_short_weights` and :func:`size_weights`); the factor is rebuilt
    from them, so passing the weights that produced it changes nothing.
    """
    active = weights.abs().sum(axis=1) > 0
    factor = (weights * returns).sum(axis=1).where(active).dropna().rename(name)
    traded = turnover(weights).reindex(factor.index).rename("turnover")
    wealth = (1.0 + factor).cumprod()
    drawdown = (wealth / wealth.cummax() - 1.0).rename("drawdown")
    break_even = float(factor.mean() / traded.mean() * 1e4)
    return Tradability(
        factor=factor,
        turnover=traded,
        drawdown=drawdown,
        break_even_bps=break_even,
    )


def momentum_tradability(
    returns: pd.DataFrame,
    *,
    n_quantiles: int = 3,
    lookback: int = 12,
    skip: int = 1,
) -> Tradability:
    """Tradability of the self-built WML spread (12-1 momentum).

    Uses :func:`momentum_signal` and :func:`long_short_weights` - the same code
    path that builds ``WML`` in :func:`build_self_factors` - so the factor here
    is exactly the reported one and the extra output is only what it costs to
    trade it.
    """
    signal = momentum_signal(returns, lookback=lookback, skip=skip)
    weights = long_short_weights(returns, signal, n_quantiles=n_quantiles, long_high=True)
    return factor_tradability(returns, weights, name="WML")


def size_tradability(
    returns: pd.DataFrame,
    value_weights: pd.DataFrame,
    *,
    n_quantiles: int = 3,
) -> Tradability:
    """Tradability of the self-built SMB spread (lagged point-in-time size)."""
    weights = size_weights(returns, value_weights, n_quantiles=n_quantiles)
    return factor_tradability(returns, weights, name="SMB")


def self_factor_tradability(
    returns: pd.DataFrame,
    value_weights: pd.DataFrame,
    *,
    n_quantiles: int = 3,
    lookback: int = 12,
    skip: int = 1,
) -> dict[str, Tradability]:
    """Tradability of both self-built factors: ``{"SMB": ..., "WML": ...}``.

    ``value_weights`` are the committed point-in-time capitalisation weights;
    both factors are built exactly as :func:`build_self_factors` builds them.
    """
    return {
        "SMB": size_tradability(returns, value_weights, n_quantiles=n_quantiles),
        "WML": momentum_tradability(
            returns, n_quantiles=n_quantiles, lookback=lookback, skip=skip
        ),
        "LOWVOL": factor_tradability(
            returns,
            low_volatility_weights(returns, n_quantiles=n_quantiles),
            name="LOWVOL",
        ),
        "REV": factor_tradability(
            returns, reversal_weights(returns, n_quantiles=n_quantiles), name="REV"
        ),
    }
