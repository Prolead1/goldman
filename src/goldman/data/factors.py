"""Factor, risk-free-rate and test-portfolio loading.

The Ken French files supply the market, size, value, profitability,
investment and momentum factors, the monthly risk-free rate, and the sorted
test portfolios.  All are cached as monthly CSVs in decimals.

``monthly_rf`` aligns the risk-free rate to an arbitrary return index and falls
back to a constant outside the factor file's coverage.  ``align_excess_returns``
puts a return panel and a factor frame on one monthly calendar and subtracts
the risk-free rate, which is the input every regression stage needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config

__all__ = [
    "load_ff3",
    "load_ff5",
    "load_momentum",
    "load_factors",
    "load_test_portfolios",
    "load_developed",
    "LongHistory",
    "load_long_history",
    "align_excess_returns",
    "monthly_rf",
    "to_period_index",
]


def _as_monthly_periods(index: pd.Index) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("M")
    return pd.PeriodIndex(pd.to_datetime(index), freq="M")


def to_period_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by monthly ``PeriodIndex`` (for safe joins)."""
    out = frame.copy()
    out.index = _as_monthly_periods(out.index)
    out.index.name = "Date"
    return out


def _read_monthly_csv(path: Path, as_decimals: bool = True) -> pd.DataFrame:
    """Read a Ken French style CSV with a ``YYYY-MM`` ``Date`` column."""
    frame = pd.read_csv(path)
    frame["Date"] = pd.PeriodIndex(pd.to_datetime(frame["Date"], format="%Y-%m"), freq="M")
    frame = frame.set_index("Date").sort_index()
    if not as_decimals:
        frame = frame / 100.0
    return frame


def load_ff3(
    path: str | Path | None = None,
    as_decimals: bool = True,
) -> pd.DataFrame:
    """Load the Fama–French 3-factor + RF file.

    The bundled file is already in decimals; ``as_decimals=False`` is provided
    for a raw Ken French CSV (values in percent).
    """
    csv_path = Path(path) if path is not None else config.data_path(config.FF3_FILENAME)
    return _read_monthly_csv(csv_path, as_decimals)


def load_ff5(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Fama–French 5-factor (2x3) file: Mkt-RF, SMB, HML, RMW, CMA, RF.

    Kept separate from :func:`load_ff3` because the five-factor SMB is built
    from different sorts, so the two SMB series are not interchangeable.
    """
    csv_path = Path(path) if path is not None else config.data_path(config.FF5_FILENAME)
    return _read_monthly_csv(csv_path)


def load_momentum(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Ken French momentum factor (``Mom``, prior 12-2 month return)."""
    csv_path = Path(path) if path is not None else config.data_path(config.MOMENTUM_FILENAME)
    return _read_monthly_csv(csv_path)


def load_factors(
    ff3_path: str | Path | None = None,
    momentum_path: str | Path | None = None,
) -> pd.DataFrame:
    """FF3 plus momentum on the months both files cover.

    Columns: ``Mkt-RF, SMB, HML, RF, Mom`` — everything the CAPM, FF3 and
    Carhart four-factor models need.
    """
    ff3 = load_ff3(ff3_path)
    mom = load_momentum(momentum_path)
    return ff3.join(mom[["Mom"]], how="inner")


def load_test_portfolios(name: str, path: str | Path | None = None) -> pd.DataFrame:
    """Load a sorted test-portfolio panel (value-weighted returns, decimals).

    ``name`` is a key into :data:`goldman.config.TEST_PORTFOLIOS`
    (``"ff25"`` for 5x5 size x book-to-market, ``"mom10"`` for momentum deciles).
    """
    if path is None:
        try:
            filename = config.TEST_PORTFOLIOS[name]
        except KeyError as exc:
            valid = ", ".join(sorted(config.TEST_PORTFOLIOS))
            raise KeyError(f"Unknown test portfolios {name!r}; choose one of: {valid}") from exc
        path = config.data_path(filename)
    return _read_monthly_csv(Path(path))


def load_developed(path: str | Path | None = None) -> pd.DataFrame:
    """Ken French developed-markets factors (USD; ``RF`` is the US T-bill).

    ``Mkt-RF`` is the value-weighted developed-markets return — an MSCI World
    analogue, used as the second market proxy for Roll's critique.
    """
    csv_path = Path(path) if path is not None else config.data_path(config.DEVELOPED_FILENAME)
    return _read_monthly_csv(csv_path)


@dataclass
class LongHistory:
    """Ken French 10 industry portfolios since 1963 plus what is needed to use them.

    ``caps`` is each industry's total market cap ($m, number of firms x average
    firm size).  Ken French measures the row for month ``t`` at the end of month
    ``t-1`` — the weights behind that month's value-weighted return — so it can
    be used directly as the weights *held during* month ``t`` (checked: these
    weights reproduce the CRSP market with a 13 bp monthly tracking error, versus
    15.5 bp when lagged a further month).  ``factors`` is the FF3 file over the
    industries' months; ``factors_full`` runs from 1926 (for the historical
    equity premium).  ``momentum`` is the Ken French momentum factor since 1927.
    """

    returns: pd.DataFrame
    caps: pd.DataFrame
    factors: pd.DataFrame
    factors_full: pd.DataFrame
    momentum: pd.DataFrame


def load_long_history(data_dir: str | Path | None = None) -> LongHistory:
    """Load the long-history robustness data set (see :data:`config.LONG_HISTORY`)."""
    base = Path(data_dir) if data_dir is not None else config.DATA_DIR
    files = config.LONG_HISTORY
    returns = _read_monthly_csv(base / files["returns"])
    n_firms = _read_monthly_csv(base / files["n_firms"])
    avg_size = _read_monthly_csv(base / files["avg_size"])
    factors = _read_monthly_csv(base / files["ff3"])
    momentum = _read_monthly_csv(base / files["momentum"])
    common = returns.index.intersection(factors.index)
    return LongHistory(
        returns=returns.loc[common],
        caps=(n_firms * avg_size).loc[common],
        factors=factors.loc[common],
        factors_full=factors,
        momentum=momentum,
    )


def align_excess_returns(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    rf_column: str = "RF",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put returns and factors on one monthly calendar and subtract the RF.

    Returns ``(excess_returns, factors)``, both indexed by the monthly
    ``PeriodIndex`` of months present in *both* inputs.  Months outside the
    factor file are dropped rather than filled, so every regression uses real
    factor and risk-free observations only.
    """
    if rf_column not in factors.columns:
        raise ValueError(f"factors must contain a {rf_column!r} column.")
    rets = to_period_index(returns)
    facs = to_period_index(factors)
    common = rets.index.intersection(facs.index)
    if len(common) == 0:
        raise ValueError("returns and factors share no months.")
    rets = rets.loc[common]
    facs = facs.loc[common]
    excess = rets.sub(facs[rf_column], axis=0)
    return excess, facs


def monthly_rf(
    index: pd.Index,
    ff3: pd.DataFrame | None = None,
    fallback: float = config.RF_MONTHLY,
) -> pd.Series:
    """Return a monthly risk-free series aligned to ``index``.

    Uses the ``RF`` column of the factor file when available and falls back to
    a constant ``fallback`` rate for dates outside its coverage (e.g. the
    synthetic sample).
    """
    if ff3 is None or "RF" not in ff3.columns:
        return pd.Series(fallback, index=index, name="RF")

    periods = pd.PeriodIndex(pd.to_datetime(index), freq="M")
    rf = ff3["RF"].reindex(periods).to_numpy()
    series = pd.Series(rf, index=index, name="RF")
    return series.fillna(fallback)