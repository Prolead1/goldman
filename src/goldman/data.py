"""Data loading and cleaning.

Reads the scraper's monthly price CSVs and the Ken French factor file, and
turns them into clean return panels.  All cleaning decisions (sparse-ticker
drops, fully-observed months, decimal conversion) are made in exactly one
place so the report can cite this module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

__all__ = [
    "load_prices",
    "prices_to_returns",
    "load_returns",
    "load_ff3",
    "monthly_rf",
    "to_period_index",
]


def _read_price_csv(path: Path) -> pd.DataFrame:
    """Read a price CSV, tolerating both indexed and unnamed ``Date`` columns."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame = frame.sort_index()
    frame.index.name = "Date"
    return frame


def load_prices(
    path: str | Path | None = None,
    universe: str | None = None,
    min_obs: int = config.MIN_OBS,
) -> pd.DataFrame:
    """Load a monthly price panel, dropping thinly-traded tickers.

    Exactly one of ``path`` or ``universe`` should be given; ``universe`` is
    a key into :data:`goldman.config.UNIVERSES`.
    """
    if (path is None) == (universe is None):
        raise ValueError("Provide exactly one of `path` or `universe`.")
    csv_path = Path(path) if path is not None else config.universe_path(universe)

    prices = _read_price_csv(csv_path)
    prices = prices.dropna(axis=1, thresh=min_obs)  # kill sparse tickers
    return prices


def prices_to_returns(
    prices: pd.DataFrame,
    min_obs: int = config.MIN_OBS,
) -> pd.DataFrame:
    """Convert a price panel to a fully-observed monthly simple-return panel."""
    returns = prices.dropna(axis=1, thresh=min_obs).pct_change()
    returns = returns.dropna(how="all")
    returns = returns.dropna(axis=0)  # keep fully-observed months
    return returns


def load_returns(
    path: str | Path | None = None,
    universe: str | None = None,
    min_obs: int = config.MIN_OBS,
) -> pd.DataFrame:
    """Convenience loader: prices in, clean monthly returns out."""
    prices = load_prices(path=path, universe=universe, min_obs=min_obs)
    return prices_to_returns(prices, min_obs=min_obs)


def to_period_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by monthly ``PeriodIndex`` (for safe joins)."""
    out = frame.copy()
    idx = pd.to_datetime(out.index)
    out.index = pd.PeriodIndex(idx, freq="M")
    out.index.name = "Date"
    return out


def load_ff3(
    path: str | Path | None = None,
    as_decimals: bool = True,
) -> pd.DataFrame:
    """Load the Fama–French 3-factor + RF file.

    The bundled file is already in decimals; ``as_decimals=False`` is provided
    for a raw Ken French CSV (values in percent).
    """
    csv_path = Path(path) if path is not None else config.data_path(config.FF3_FILENAME)
    frame = pd.read_csv(csv_path)
    frame["Date"] = pd.PeriodIndex(pd.to_datetime(frame["Date"], format="%Y-%m"), freq="M")
    frame = frame.set_index("Date").sort_index()
    if not as_decimals:
        frame = frame / 100.0
    return frame


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