"""A self-built factor: momentum as a long–short spread portfolio.

The handout credits groups that build a factor themselves rather than download
it.  Momentum needs only returns, so it can be built from our own data:

* **signal** — each asset's cumulative return from month ``t-12`` to ``t-2``,
  skipping month ``t-1`` (short-term reversal), known at the end of ``t-1``;
* **portfolio** — each month, equal-weight long the ``n`` assets with the
  highest signal and short the ``n`` with the lowest; hold one month;
* **validation** — correlation with Ken French's momentum factor (built from
  every CRSP stock), and whether its average return survives the CAPM and FF3
  and a transaction cost charged on turnover.

On the 10 Ken French industries since 1963 this is industry momentum
(Moskowitz and Grinblatt 1999), with 60 years of data; on our 30 assets it is a
short, noisy illustration only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from .regression import time_series_regression

__all__ = [
    "SelfBuiltFactor",
    "momentum_signal",
    "long_short_factor",
    "factor_report",
]

ANN = config.TRADING_MONTHS


def momentum_signal(returns: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """Cumulative return over months ``t-lookback .. t-1-skip`` for each month ``t``.

    The value in row ``t`` uses only returns up to ``t-1-skip``, so it is known
    when the portfolio for month ``t`` is formed.
    """
    window = lookback - skip
    growth = (1.0 + returns).shift(skip + 1)
    return growth.rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0


@dataclass
class SelfBuiltFactor:
    """Monthly factor returns, the weights behind them, and rebalancing turnover."""

    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series


def long_short_factor(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    n_side: int,
    name: str = "factor",
) -> SelfBuiltFactor:
    """Equal-weight long the top ``n_side`` and short the bottom ``n_side`` by signal.

    Months where fewer than ``2 * n_side`` assets have a signal are skipped.
    Turnover is the sum of absolute weight changes between months (the drift of
    weights within a month is ignored).
    """
    ranks = signal.rank(axis=1, method="first")
    counts = signal.notna().sum(axis=1)
    usable = counts >= 2 * n_side
    top = ranks.gt(counts - n_side, axis=0) & signal.notna()
    bottom = ranks.le(n_side) & signal.notna()
    weights = (top.astype(float) - bottom.astype(float)) / n_side
    weights = weights.loc[usable]
    factor = (weights * returns.loc[weights.index]).sum(axis=1).rename(name)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = np.nan  # setting up the first portfolio is not turnover
    return SelfBuiltFactor(returns=factor, weights=weights, turnover=turnover.rename(name))


def factor_report(
    built: SelfBuiltFactor,
    factors: pd.DataFrame,
    reference: pd.Series | None = None,
    cost_bps: float = 50.0,
) -> pd.Series:
    """Is the self-built factor real, priced, and worth trading?

    ``factors`` must hold ``Mkt-RF``, ``SMB`` and ``HML`` on (at least) the
    factor's months; ``reference`` is the downloaded factor it should resemble.
    A long–short return is already an excess return, so it is regressed
    directly.  ``net`` subtracts ``cost_bps`` per unit of turnover.
    """
    f = built.returns
    common = f.index.intersection(factors.index)
    f = f.loc[common]
    regs = factors.loc[common]
    capm = time_series_regression(f.to_frame(), regs[["Mkt-RF"]], hac_lags=0)
    ff3 = time_series_regression(f.to_frame(), regs[["Mkt-RF", "SMB", "HML"]], hac_lags=0)
    net = f - (cost_bps / 1e4) * built.turnover.reindex(common).fillna(0.0)
    n = len(f)
    out = {
        "first month": str(common[0]),
        "months": n,
        "mean (ann.)": f.mean() * ANN,
        "t(mean)": f.mean() / (f.std() / np.sqrt(n)),
        "sharpe (ann.)": f.mean() / f.std() * np.sqrt(ANN),
        "CAPM alpha (ann.)": float(capm.alpha.iloc[0]) * ANN,
        "t(CAPM alpha)": float(capm.tstats["alpha"].iloc[0]),
        "FF3 alpha (ann.)": float(ff3.alpha.iloc[0]) * ANN,
        "t(FF3 alpha)": float(ff3.tstats["alpha"].iloc[0]),
        "turnover (monthly)": float(built.turnover.mean()),
        f"mean net of {cost_bps:g}bp (ann.)": net.mean() * ANN,
        f"t(net of {cost_bps:g}bp)": net.mean() / (net.std() / np.sqrt(n)),
        # Cost per unit of turnover at which the average return falls to zero.
        "break-even cost (bp)": float(f.mean() / built.turnover.mean() * 1e4),
    }
    if reference is not None:
        ref = reference.reindex(common)
        out["corr with Ken French Mom"] = float(f.corr(ref))
    return pd.Series(out, name=built.returns.name)
