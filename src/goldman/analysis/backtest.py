"""Stage 4: an out-of-sample horse race between portfolio rules.

Every month ``t`` each rule sees only the previous ``window`` months of excess
returns (and the market weights known at the end of month ``t-1``), forms
weights, and earns month ``t``'s return.  This is the DeMiguel, Garlappi and
Uppal (2009) design: it asks which rule an investor could actually have
followed, so estimation error is paid for rather than assumed away.

Rules (all fixed before looking at any result):

* ``value weighted`` — the market portfolio of the universe;
* ``1/N`` — equal weights;
* ``min variance`` — sample covariance, unconstrained;
* ``tangency (plug-in)`` — ``Sigma^-1 mu`` from sample moments;
* ``tangency (long-only)`` — the same objective with no short sales;
* ``tangency (shrinkage)`` — Ledoit–Wolf covariance and means shrunk toward
  the CAPM prior with a James–Stein intensity: the "something in between";
* ``tangency (shrinkage, long-only)`` — the same inputs with no short sales.

The CAPM prior needs a market premium.  It is the historical US equity premium
*known at the decision date* (expanding mean of the Ken French market excess
return since 1926), not the 60-month window mean: an equilibrium premium is
positive, while a window mean turns negative after bear markets and would make
the prior short the market.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; imported before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .. import config  # noqa: E402
from .estimators import capm_prior_means, ledoit_wolf_covariance, shrink_means  # noqa: E402
from .portfolio import (  # noqa: E402
    min_variance_weights,
    tangency_weights_long_only,
)

__all__ = [
    "Window",
    "BacktestResult",
    "STANDARD_RULES",
    "value_weighted",
    "equal_weighted",
    "min_variance",
    "tangency_plugin",
    "tangency_long_only",
    "tangency_shrinkage",
    "tangency_shrinkage_long_only",
    "historical_premium",
    "tangency_sign_flips",
    "holding_weights",
    "rolling_backtest",
    "performance_table",
    "sharpe_difference_test",
    "plot_cumulative",
]

ANN = config.TRADING_MONTHS
COST_BPS = 50  # proportional cost per unit of turnover (DeMiguel et al. 2009)

_STYLE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ------------------------------------------------------------------ rules
@dataclass
class Window:
    """What a rule may see when choosing weights for month ``t``.

    ``excess`` holds months ``t-window .. t-1``.  ``market_weights`` are the
    value weights known at the end of month ``t-1``; ``market_excess`` is the
    value-weighted portfolio's excess return over the same window;
    ``prior_premium`` is the equilibrium market premium (monthly) known then.
    """

    excess: pd.DataFrame
    market_weights: pd.Series | None = None
    market_excess: pd.Series | None = None
    prior_premium: float | None = None


Rule = Callable[[Window], pd.Series]


def _sum_to_one(raw: np.ndarray, index: pd.Index) -> pd.Series:
    """Normalise by ``|1'x|`` so a negative sum does not flip every position."""
    return pd.Series(raw / abs(raw.sum()), index=index)


def value_weighted(win: Window) -> pd.Series:
    if win.market_weights is None:
        raise ValueError("value_weighted needs market weights.")
    w = win.market_weights.reindex(win.excess.columns)
    return w / w.sum()


def equal_weighted(win: Window) -> pd.Series:
    n = win.excess.shape[1]
    return pd.Series(1.0 / n, index=win.excess.columns)


def min_variance(win: Window) -> pd.Series:
    return min_variance_weights(win.excess.cov())


def tangency_plugin(win: Window) -> pd.Series:
    raw = np.linalg.solve(win.excess.cov().to_numpy(), win.excess.mean().to_numpy())
    return _sum_to_one(raw, win.excess.columns)


def tangency_long_only(win: Window) -> pd.Series:
    return tangency_weights_long_only(win.excess.mean(), win.excess.cov())


def _shrinkage_inputs(win: Window) -> tuple[pd.Series, pd.DataFrame]:
    """Ledoit–Wolf covariance and means shrunk toward the CAPM prior."""
    if win.market_weights is None:
        raise ValueError("shrinkage rules need market weights.")
    if win.prior_premium is not None:
        premium = win.prior_premium
    elif win.market_excess is not None:
        premium = float(win.market_excess.mean())
    else:
        raise ValueError("shrinkage rules need a prior premium or market returns.")
    cov, _ = ledoit_wolf_covariance(win.excess)
    w_m = win.market_weights.reindex(win.excess.columns)
    prior = capm_prior_means(cov, w_m / w_m.sum(), premium)
    means, _ = shrink_means(win.excess.mean(), prior, cov, len(win.excess))
    return means, cov


def tangency_shrinkage(win: Window) -> pd.Series:
    """Ledoit–Wolf covariance, means shrunk toward the CAPM prior.

    With full shrinkage this *is* the value-weighted portfolio; with none it is
    the tangency portfolio on the Ledoit–Wolf covariance.  The data choose the
    blend through the James–Stein intensity.
    """
    means, cov = _shrinkage_inputs(win)
    raw = np.linalg.solve(cov.to_numpy(), means.to_numpy())
    return _sum_to_one(raw, win.excess.columns)


def tangency_shrinkage_long_only(win: Window) -> pd.Series:
    """The shrinkage inputs with no short sales (constraints shrink further)."""
    means, cov = _shrinkage_inputs(win)
    return tangency_weights_long_only(means, cov)


STANDARD_RULES: dict[str, Rule] = {
    "value weighted": value_weighted,
    "1/N": equal_weighted,
    "min variance": min_variance,
    "tangency (plug-in)": tangency_plugin,
    "tangency (long-only)": tangency_long_only,
    "tangency (shrinkage)": tangency_shrinkage,
    "tangency (shrinkage, long-only)": tangency_shrinkage_long_only,
}


def historical_premium(market_excess: pd.Series, months: pd.Index) -> pd.Series:
    """Expanding mean of the market excess return known at each decision date.

    The value for month ``t`` averages every month *before* ``t`` (no
    look-ahead).  ``market_excess`` should start well before ``months`` (the
    Ken French file runs from 1926).
    """
    known = market_excess.sort_index().expanding().mean().shift(1)
    return known.reindex(months)


def tangency_sign_flips(excess_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """How often the rolling plug-in tangency portfolio is ill-defined.

    For each window, ``1' Sigma^-1 mu`` is the denominator that scales the
    tangency weights to sum to one.  When it is negative the "tangency"
    portfolio sits on the inefficient half of the frontier; when it is near zero
    the weights explode.  With ``T / N`` close to one both happen often — the
    sharpest symptom of estimation error.  Returns one row per window with the
    denominator and the resulting gross leverage.
    """
    rows = {}
    for t in range(window, len(excess_returns) + 1):
        win = excess_returns.iloc[t - window : t]
        raw = np.linalg.solve(win.cov().to_numpy(), win.mean().to_numpy())
        rows[win.index[-1]] = {
            "denominator": float(raw.sum()),
            "gross leverage": float(np.abs(raw).sum() / abs(raw.sum())),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "window end"
    return out


# ----------------------------------------------------------- the engine
def holding_weights(month_end_weights: pd.DataFrame) -> pd.DataFrame:
    """Turn month-end market weights into the weights *held during* each month.

    Row ``t`` of the result is the row for ``t-1`` of the input, renormalised:
    a portfolio formed at the end of month ``t-1`` earns month ``t``'s return.
    The first month has no prior weights and is dropped.
    """
    held = month_end_weights.shift(1).iloc[1:]
    return held.div(held.sum(axis=1), axis=0)


@dataclass
class BacktestResult:
    """Out-of-sample excess returns, weights and turnover for every rule."""

    returns: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    turnover: pd.DataFrame
    window: int


def rolling_backtest(
    excess_returns: pd.DataFrame,
    rules: dict[str, Rule],
    window: int,
    held_market_weights: pd.DataFrame | None = None,
    rf: pd.Series | None = None,
    prior_premium: pd.Series | None = None,
) -> BacktestResult:
    """Run every rule on a rolling ``window`` and record its next-month return.

    ``held_market_weights`` (see :func:`holding_weights`) must share the index
    of ``excess_returns``: row ``t`` is what a value-weighted investor holds in
    month ``t``.  ``rf`` (per month) is used only to drift weights between
    rebalancing dates when measuring turnover.  ``prior_premium`` (see
    :func:`historical_premium`) is the market premium the shrinkage rules use
    for month ``t``; it must already be lagged.
    """
    if window >= len(excess_returns):
        raise ValueError("window must be shorter than the sample.")
    if held_market_weights is not None:
        if not held_market_weights.index.equals(excess_returns.index):
            raise ValueError("held_market_weights must share the excess_returns index.")
        held_market_weights = held_market_weights[excess_returns.columns]
        mkt_series = (held_market_weights * excess_returns).sum(axis=1)
    else:
        mkt_series = None
    rf_values = (
        rf.reindex(excess_returns.index).fillna(0.0)
        if rf is not None
        else pd.Series(0.0, index=excess_returns.index)
    )

    dates = excess_returns.index[window:]
    weights: dict[str, list[pd.Series]] = {name: [] for name in rules}
    for t in range(window, len(excess_returns)):
        win = Window(
            excess=excess_returns.iloc[t - window : t],
            market_weights=None if held_market_weights is None else held_market_weights.iloc[t],
            market_excess=None if mkt_series is None else mkt_series.iloc[t - window : t],
            prior_premium=None
            if prior_premium is None
            else float(prior_premium.loc[excess_returns.index[t]]),
        )
        for name, rule in rules.items():
            weights[name].append(rule(win).reindex(excess_returns.columns).fillna(0.0))

    oos = excess_returns.loc[dates]
    total = oos.add(rf_values.loc[dates], axis=0)
    out_returns, out_weights, out_turnover = {}, {}, {}
    for name, rows in weights.items():
        w = pd.DataFrame(rows, index=dates)
        out_weights[name] = w
        out_returns[name] = (w * oos).sum(axis=1)
        # Weights drift with returns between rebalancing dates.
        grown = w * (1.0 + total)
        drifted = grown.div(grown.sum(axis=1), axis=0).shift(1)
        turnover = (w - drifted).abs().sum(axis=1)
        turnover.iloc[0] = np.nan  # initial purchase is not turnover
        out_turnover[name] = turnover

    return BacktestResult(
        returns=pd.DataFrame(out_returns),
        weights=out_weights,
        turnover=pd.DataFrame(out_turnover),
        window=window,
    )


# ------------------------------------------------------------ evaluation
def sharpe_difference_test(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Jobson–Korkie test of equal Sharpe ratios with Memmel's (2003) correction.

    Returns ``(z, two-sided p-value)``; inputs are monthly excess returns.
    """
    a, b = a.align(b, join="inner")
    n = len(a)
    sr_a, sr_b = a.mean() / a.std(), b.mean() / b.std()
    rho = a.corr(b)
    var = (2 - 2 * rho + 0.5 * (sr_a**2 + sr_b**2 - 2 * sr_a * sr_b * rho**2)) / n
    if var <= 0:
        return 0.0, 1.0
    z = (sr_a - sr_b) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def _max_drawdown(r: pd.Series) -> float:
    wealth = (1.0 + r).cumprod()
    if (wealth <= 0).any():
        return -1.0
    return float((wealth / wealth.cummax() - 1.0).min())


def performance_table(
    result: BacktestResult,
    benchmark: str = "value weighted",
    references: dict[str, pd.Series] | None = None,
    cost_bps: float = COST_BPS,
    risk_aversions: tuple[float, ...] = (1.0, 5.0),
) -> pd.DataFrame:
    """One row per rule (and per reference index): the horse-race scorecard.

    Sharpe ratios are annualised; ``p vs <benchmark>`` is the Jobson–Korkie–
    Memmel test of equal Sharpe ratios.  ``sharpe net`` subtracts
    ``cost_bps`` per unit of turnover.  References (e.g. the CRSP index) are
    buy-and-hold, so they carry no turnover.  ``CEQ g=..`` is the certainty
    equivalent excess return of a mean–variance investor with that risk
    aversion, ``12 * (mean - g/2 * var)`` (DeMiguel et al. use ``g = 1``): the
    riskless return the investor would accept instead of the rule.
    """
    returns = result.returns.copy()
    if references:
        for name, series in references.items():
            returns[name] = series.reindex(returns.index)
    bench = returns[benchmark]
    rows = {}
    for name in returns.columns:
        r = returns[name]
        turnover = result.turnover[name] if name in result.turnover else None
        mean_turn = float(turnover.mean()) if turnover is not None else 0.0
        net = r - (cost_bps / 1e4) * (turnover.fillna(0.0) if turnover is not None else 0.0)
        weights = result.weights.get(name)
        gross = float(weights.abs().sum(axis=1).mean()) if weights is not None else 1.0
        biggest = float(weights.abs().max(axis=1).mean()) if weights is not None else np.nan
        z, p = sharpe_difference_test(r, bench) if name != benchmark else (0.0, 1.0)
        rows[name] = {
            "mean (ann.)": r.mean() * ANN,
            "vol (ann.)": r.std() * np.sqrt(ANN),
            "sharpe (ann.)": r.mean() / r.std() * np.sqrt(ANN),
            f"p vs {benchmark}": p,
            f"sharpe net {cost_bps:g}bp": net.mean() / net.std() * np.sqrt(ANN),
            "max drawdown": _max_drawdown(r),
            "worst month": r.min(),
            "turnover (monthly)": mean_turn,
            "gross leverage": gross,
            "max |weight|": biggest,
            **{
                f"CEQ g={g:g} (ann.)": ANN * (r.mean() - g / 2.0 * r.var())
                for g in risk_aversions
            },
        }
    out = pd.DataFrame(rows).T
    out.index.name = "rule"
    return out


def plot_cumulative(
    returns: pd.DataFrame,
    output_path: str | Path,
    title: str = "Out-of-sample growth of $1 (excess returns)",
    exclude: list[str] | None = None,
    dashed: list[str] | None = None,
) -> Path:
    """Cumulative excess-return wealth of each rule on a log scale.

    Rules whose wealth hits zero cannot be drawn on a log axis; pass them in
    ``exclude`` and name them in the caption instead.  ``dashed`` series (e.g.
    an index used as a reference) are drawn dashed so a rule that tracks them
    stays visible.
    """
    output_path = Path(output_path)
    cols = [c for c in returns.columns if c not in (exclude or [])]
    if len(cols) > len(_STYLE):
        raise ValueError(f"At most {len(_STYLE)} series fit on one chart.")
    wealth = (1.0 + returns[cols]).cumprod()
    x = wealth.index.to_timestamp() if isinstance(wealth.index, pd.PeriodIndex) else wealth.index

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.grid(color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for name, color in zip(cols, _STYLE, strict=False):
        style = "--" if name in (dashed or []) else "-"
        ax.plot(x, wealth[name], color=color, lw=2, ls=style, label=name)
    ax.set_yscale("log")
    ax.axhline(1.0, color=_MUTED, lw=0.8)
    ax.set_ylabel("wealth (log scale)", color=_INK)
    ax.set_title(title, color=_INK, loc="left")
    ax.tick_params(colors=_MUTED)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
