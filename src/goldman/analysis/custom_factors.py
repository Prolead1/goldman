"""Stage 3 extension — two factors of our own, run on both markets.

Momentum is the one self-built factor that earns a premium in these
universes, so both custom factors refine it rather than search elsewhere:

* **52-week high (NEAR52).** George and Hwang (2004): investors anchor on a
  stock's 52-week high and under-react to news that pushes the price towards
  it, so stocks trading close to their high keep outperforming.  Each month the
  signal is the asset's index level at the end of ``t-1`` divided by its highest
  month-end level over ``t-12`` to ``t-1``.  We go long the third closest to its
  high and short the third furthest away, equal-weighted (the same 30/40/30
  convention as SMB and WML).  The index is the total-return series both
  markets share, not the quoted price George and Hwang use.
* **Volatility-managed momentum (VWML).** Barroso and Santa-Clara (2015):
  momentum's risk is predictable and its crashes come in high-volatility
  spells, so scaling the position by its recent volatility keeps the premium
  and cuts the crashes.  Each month WML is scaled by
  ``12% / (annualised std of WML over t-6 .. t-1)``, a rule known before the
  month it trades.

Both specifications are the papers' defaults (30/40/30 breakpoints; a
six-month window and a 12% target), not tuned to our data;
``scripts/candidate_factors.py`` reports them beside every other signal tried.  The
scorecard reports the market-adjusted performance of each, and the spanning
regressions ask the questions that matter: is NEAR52 priced beyond the market,
size and momentum, and does scaling add alpha over plain WML?
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .factor_construction import long_short_weights, momentum_signal
from .factor_models import factor_regression

__all__ = [
    "near_high_signal",
    "near_high_weights",
    "volatility_scale",
    "turnover",
    "max_drawdown",
    "factor_scorecard",
    "spanning_regression",
    "build_custom_factors",
]

def near_high_signal(returns: pd.DataFrame, *, window: int = 12) -> pd.DataFrame:
    """Index level at ``t-1`` over its highest month-end level in ``t-window .. t-1``.

    A value of 1 means the asset closed ``t-1`` at its 12-month high.  Every
    input is realised before month ``t``, so the signal carries no look-ahead.
    """
    if window < 2:
        raise ValueError("window must be at least 2 months.")
    level = (1.0 + returns).cumprod().shift(1)
    high = level.rolling(window=window, min_periods=window).max()
    return level / high


def near_high_weights(
    returns: pd.DataFrame, *, window: int = 12, n_quantiles: int = 3
) -> pd.DataFrame:
    """NEAR52 holdings: long the names closest to their high, short the furthest."""
    signal = near_high_signal(returns, window=window)
    return long_short_weights(returns, signal, n_quantiles=n_quantiles, long_high=True)


def volatility_scale(
    factor: pd.Series,
    *,
    window: int = 6,
    target_vol: float = 0.12,
    periods_per_year: int = 12,
) -> pd.Series:
    """Position size ``target / sigma_hat`` with ``sigma_hat`` from ``t-window .. t-1``.

    ``sigma_hat`` is the annualised standard deviation of the factor's own
    monthly returns over the previous ``window`` months, so month ``t``'s size is
    set before month ``t`` begins.  No cap is applied, as in Barroso and
    Santa-Clara.
    """
    if window < 2:
        raise ValueError("window must be at least 2 months.")
    sigma = factor.shift(1).rolling(window=window, min_periods=window).std()
    return (target_vol / (sigma * np.sqrt(periods_per_year))).rename("scale")


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


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the compounded series (a negative number)."""
    growth = (1.0 + returns.dropna()).cumprod()
    peak = growth.cummax().clip(lower=1.0)
    return float((growth / peak - 1.0).min())


def factor_scorecard(
    factors: pd.DataFrame,
    market: pd.Series,
    *,
    turnovers: pd.DataFrame | None = None,
    cost_bps: float = 50.0,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """One row per factor on the sample where every factor and the market exist.

    Reports the annualised mean, volatility and Sharpe ratio (with its standard
    error under i.i.d. returns), the t-statistic of the mean, the CAPM alpha and
    beta against ``market``, the maximum drawdown and, when ``turnovers`` is
    given, monthly turnover, the net mean after ``cost_bps`` per unit traded and
    the break-even cost that would wipe out the gross premium.
    """
    panel = pd.concat([factors, market.rename("Mkt-RF")], axis=1).dropna()
    sample = panel[factors.columns]
    n = len(sample)
    capm = factor_regression(sample, panel[["Mkt-RF"]], rf=0.0)

    rows = {}
    for name in factors.columns:
        series = sample[name]
        mean, std = series.mean(), series.std()
        sharpe = mean / std
        row = {
            "months": n,
            "ann. mean": mean * periods_per_year,
            "ann. vol": std * np.sqrt(periods_per_year),
            "Sharpe": sharpe * np.sqrt(periods_per_year),
            "Sharpe se": np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(periods_per_year),
            "t(mean)": mean / (std / np.sqrt(n)),
            "CAPM alpha (ann.)": capm.loc[name, "alpha"] * periods_per_year,
            "t(CAPM alpha)": capm.loc[name, "t_alpha"],
            "market beta": capm.loc[name, "beta_Mkt-RF"],
            "max drawdown": max_drawdown(series),
        }
        if turnovers is not None and name in turnovers:
            traded = turnovers[name].reindex(sample.index)
            cost = traded * cost_bps / 1e4
            row["turnover / month"] = traded.mean()
            row[f"net mean at {cost_bps:g}bp (ann.)"] = (series - cost).mean() * periods_per_year
            row["break-even cost (bp)"] = mean / traded.mean() * 1e4
        rows[name] = row
    table = pd.DataFrame(rows).T
    table.index.name = "factor"
    return table


def spanning_regression(
    tests: Mapping[str, tuple[pd.Series, pd.DataFrame]],
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Regress each new factor on the factors it must beat; alpha is the test.

    ``tests`` maps a label to ``(factor, controls)``.  A significant positive
    alpha says the new factor earns something the controls do not.
    """
    rows = {}
    for label, (factor, controls) in tests.items():
        frame = pd.concat([factor.rename("factor"), controls], axis=1).dropna()
        table = factor_regression(frame[["factor"]], frame[controls.columns], rf=0.0)
        row = table.loc["factor"]
        rows[label] = {
            "months": int(row["n"]),
            "alpha (ann.)": row["alpha"] * periods_per_year,
            "t(alpha)": row["t_alpha"],
            "R2": row["r_squared"],
            **{f"beta {name}": row[f"beta_{name}"] for name in controls.columns},
        }
    return pd.DataFrame(rows).T


def build_custom_factors(
    returns: pd.DataFrame,
    *,
    window_high: int = 12,
    n_quantiles: int = 3,
    vol_window: int = 6,
    target_vol: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """WML, NEAR52 and VWML for one universe, with each one's monthly turnover.

    Returns ``(factors, turnovers)``.  WML is rebuilt here from its holdings (it
    matches :func:`goldman.analysis.factor_construction.momentum_factor`) so its
    turnover, and VWML's, can be measured.  VWML's holdings are WML's scaled by
    the month's position size, so its turnover counts the resizing as well.
    """
    wml_weights = long_short_weights(
        returns, momentum_signal(returns), n_quantiles=n_quantiles, long_high=True
    )
    near_weights = near_high_weights(returns, window=window_high, n_quantiles=n_quantiles)

    def spread(weights: pd.DataFrame) -> pd.Series:
        value = (weights * returns).sum(axis=1)
        return value.where(weights.abs().sum(axis=1) > 0)

    wml = spread(wml_weights)
    scale = volatility_scale(wml, window=vol_window, target_vol=target_vol)
    vwml_weights = wml_weights.mul(scale, axis=0)
    vwml = spread(vwml_weights)

    factors = pd.DataFrame({"WML": wml, "NEAR52": spread(near_weights), "VWML": vwml})
    turnovers = pd.DataFrame(
        {
            "WML": turnover(wml_weights),
            "NEAR52": turnover(near_weights),
            "VWML": turnover(vwml_weights.fillna(0.0)),
        }
    )
    factors.index.name = turnovers.index.name = returns.index.name
    return factors, turnovers
