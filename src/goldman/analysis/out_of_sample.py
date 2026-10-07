"""Out-of-sample validation of the portfolio constructions.

Stage 1 and the robustness layer compare portfolio *weights*.  This module asks
the question an investor actually faces: if the weights are estimated from data
up to month ``t-1`` and held through month ``t``, what does the realised
excess-return path look like?  Every strategy is re-estimated each month on an
expanding window and scored only on the month that follows, so no weights ever
see the return they are traded on.  The value-weighted index and 1/N need no
estimation at all and act as the honest benchmarks against which the estimated
portfolios have to earn their place.

The result is a set of monthly out-of-sample excess returns per strategy, their
cumulative growth, and a Sharpe (and drawdown) summary, plus the two exhibits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..viz import plot_oos_growth, plot_oos_sharpe
from .portfolio import equal_weights
from .robustness import VARIANTS, variant_weights

__all__ = [
    "OOS_BENCHMARKS",
    "OutOfSampleResult",
    "walk_forward_returns",
    "out_of_sample_summary",
    "run_out_of_sample_analysis",
]

#: Estimator-free benchmarks that need no fitting window.
OOS_BENCHMARKS = ("index", "equal_weight")


def _align_rf(index: pd.Index, rf: float | pd.Series) -> float | pd.Series:
    return rf.reindex(index) if isinstance(rf, pd.Series) else rf


def _rf_value(rf: float | pd.Series, date) -> float:
    if isinstance(rf, pd.Series):
        return float(rf.loc[date])
    return float(rf)


def walk_forward_returns(
    returns: pd.DataFrame,
    rf: float | pd.Series = config.RF_MONTHLY,
    *,
    min_train: int = 60,
    variants: tuple[str, ...] = VARIANTS,
    index_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Monthly out-of-sample excess returns, one column per strategy.

    ``returns`` before ``min_train`` is the first training window.  For every
    later month the construction is re-estimated on the expanding window that
    ends the month before, then applied to that month's returns.  ``index`` is
    the supplied value-weighted index excess return (or omitted); ``equal_weight``
    is the 1/N portfolio, rebalanced monthly.
    """
    if min_train < 2:
        raise ValueError("min_train must be at least 2.")
    if min_train >= len(returns):
        raise ValueError("min_train must leave at least one out-of-sample month.")

    dates = returns.index[min_train:]
    columns: dict[str, pd.Series] = {}

    if index_returns is not None:
        columns["index"] = index_returns.reindex(dates).astype(float)

    equal = (returns @ equal_weights(returns.columns)).reindex(dates)
    columns["equal_weight"] = equal - [_rf_value(rf, date) for date in dates]

    for variant in variants:
        path: list[float] = []
        for position in range(min_train, len(returns)):
            train = returns.iloc[:position]
            weights = variant_weights(
                train, _align_rf(train.index, rf), variant
            )
            realised = float(returns.iloc[position] @ weights)
            path.append(realised - _rf_value(rf, returns.index[position]))
        columns[variant] = pd.Series(path, index=dates, name=variant)

    out = pd.DataFrame(columns)
    out.index.name = "Date"
    return out


def out_of_sample_summary(oos_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised mean, volatility, Sharpe and worst drawdown per strategy."""
    growth = (1.0 + oos_returns).cumprod()
    drawdown = growth / growth.cummax() - 1.0
    rows = {}
    for name in oos_returns.columns:
        series = oos_returns[name].dropna()
        deviation = float(series.std())
        rows[name] = {
            "months": int(series.count()),
            "ann. mean": float(series.mean() * config.TRADING_MONTHS),
            "ann. vol": float(deviation * np.sqrt(config.TRADING_MONTHS)),
            "Sharpe": float(series.mean() / deviation * np.sqrt(config.TRADING_MONTHS))
            if deviation > 0
            else np.nan,
            "max drawdown": float(drawdown[name].min()),
        }
    return pd.DataFrame(rows).T


@dataclass
class OutOfSampleResult:
    """Monthly paths and exhibits produced by :func:`run_out_of_sample_analysis`."""

    universe: str
    returns: pd.DataFrame
    growth: pd.DataFrame
    summary: pd.DataFrame
    output_paths: dict[str, Path]


def run_out_of_sample_analysis(
    returns: pd.DataFrame,
    *,
    universe: str,
    rf: float | pd.Series = config.RF_MONTHLY,
    index_returns: pd.Series | None = None,
    min_train: int = 60,
    variants: tuple[str, ...] = VARIANTS,
    output_dir: str | Path | None = None,
    label: str = "us",
) -> OutOfSampleResult:
    """Run the expanding-window backtest and write the two exhibits.

    Writes ``out_of_sample_<label>_growth.png`` and
    ``out_of_sample_<label>_sharpe.png`` and returns the paths alongside the
    monthly return and growth panels.
    """
    out_dir = config.ensure_outputs_dir(Path(output_dir) if output_dir else None)
    monthly = walk_forward_returns(
        returns,
        rf,
        min_train=min_train,
        variants=variants,
        index_returns=index_returns,
    )
    growth = (1.0 + monthly).cumprod()
    summary = out_of_sample_summary(monthly).round(4)

    growth_path = plot_oos_growth(
        growth,
        out_dir / f"out_of_sample_{label}_growth.png",
        title=f"Out-of-sample growth of 1 unit ({label.upper()})",
    )
    sharpe_path = plot_oos_sharpe(
        summary,
        out_dir / f"out_of_sample_{label}_sharpe.png",
        title=f"Out-of-sample Sharpe by strategy ({label.upper()})",
    )
    return OutOfSampleResult(
        universe=universe,
        returns=monthly,
        growth=growth,
        summary=summary,
        output_paths={"growth": growth_path, "sharpe": sharpe_path},
    )
