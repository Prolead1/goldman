"""Industry momentum on sixty years of US data (report, after the memo).

The memo argues that our 30 survivors over ten years are the wrong instrument
for testing a factor.  This module runs momentum on the instrument we lack:
Ken French's 10 value-weighted industry portfolios, which hold every CRSP
stock (so nothing is missing for having failed) and start in 1963-07.

* **Signal and portfolio.** Moskowitz and Grinblatt (1999): each month rank the
  industries on their return from ``t-12`` to ``t-2``, go long the best three
  and short the worst three, equal-weighted, and hold one month.  It is the WML
  rule of Section 5 applied to industries (``10 // 3 = 3`` names per leg), so the
  first factor month is 1964-07.
* **Tests.** The mean and Sharpe ratio; alphas against the CAPM and FF3; the
  correlation with Ken French's stock-level momentum factor ``Mom``; and the
  cost of trading it (50 bp per unit of turnover, and the break-even cost).

:func:`industry_momentum_report` reports these over the full sample and over
sub-periods, so the decade our own data cover can be read against the sixty
years before it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .factor_construction import long_short_weights, momentum_signal
from .factor_models import factor_regression

__all__ = ["build_industry_momentum", "industry_momentum_report", "turnover"]


def _timestamps(frame):
    """Month-start timestamps for a Ken French ``PeriodIndex`` (regressions need them)."""
    if isinstance(frame.index, pd.PeriodIndex):
        frame = frame.copy()
        frame.index = frame.index.to_timestamp()
    return frame


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


def build_industry_momentum(
    industries: pd.DataFrame,
    *,
    n_quantiles: int = 3,
    lookback: int = 12,
    skip: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """Industry momentum returns and monthly turnover, from the first full month.

    Returns ``(factor, traded)`` on a month-start ``DatetimeIndex``.  Turnover is
    the sum of absolute weight changes; the first month counts the full gross
    position of 2 (see :func:`turnover`).
    """
    industries = _timestamps(industries)
    signal = momentum_signal(industries, lookback=lookback, skip=skip)
    weights = long_short_weights(industries, signal, n_quantiles=n_quantiles, long_high=True)
    active = weights.abs().sum(axis=1) > 0
    factor = (weights * industries).sum(axis=1).where(active).dropna().rename("IndMom")
    return factor, turnover(weights).reindex(factor.index).rename("IndMom")


def industry_momentum_report(
    factor: pd.Series,
    traded: pd.Series,
    factors: pd.DataFrame,
    *,
    periods: Mapping[str, tuple[str, str]],
    cost_bps: float = 50.0,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """One row per period: premium, CAPM and FF3 alphas, link to ``Mom``, costs.

    ``factors`` holds ``Mkt-RF``, ``SMB``, ``HML`` and ``Mom`` (a long-short
    return is already an excess return, so no risk-free rate is taken off).
    ``periods`` maps a label to an inclusive ``(start, end)`` pair of months.
    """
    factors = _timestamps(factors)
    ann = periods_per_year
    rows = {}
    for label, (start, end) in periods.items():
        f = factor.loc[start:end].dropna()
        cost = traded.reindex(f.index)
        frame = f.to_frame("f")
        capm = factor_regression(frame, factors[["Mkt-RF"]], rf=0.0).loc["f"]
        ff3 = factor_regression(frame, factors[["Mkt-RF", "SMB", "HML"]], rf=0.0).loc["f"]
        n, mean, std = len(f), f.mean(), f.std()
        rows[label] = {
            "months": n,
            "ann. mean": mean * ann,
            "t(mean)": mean / (std / np.sqrt(n)),
            "Sharpe": mean / std * np.sqrt(ann),
            "CAPM alpha (ann.)": capm["alpha"] * ann,
            "t(CAPM alpha)": capm["t_alpha"],
            "FF3 alpha (ann.)": ff3["alpha"] * ann,
            "t(FF3 alpha)": ff3["t_alpha"],
            "corr with Mom": float(f.corr(factors["Mom"].reindex(f.index))),
            "turnover / month": cost.mean(),
            f"net mean at {cost_bps:g}bp (ann.)": (f - cost * cost_bps / 1e4).mean() * ann,
            "break-even cost (bp)": mean / cost.mean() * 1e4,
        }
    table = pd.DataFrame(rows).T
    table.index.name = "period"
    return table
