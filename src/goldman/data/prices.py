"""Price loading and the return panel used by every analysis.

Reads the scraper's monthly price CSVs and turns them into clean monthly
simple-return panels.  All cleaning decisions (sparse-ticker drops,
fully-observed months) are made here so the report can cite one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config

__all__ = [
    "load_prices",
    "prices_to_returns",
    "load_returns",
    "load_index_proxy_returns",
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

def load_index_proxy_returns(path: str | Path | None = None) -> pd.DataFrame:
    """Monthly total returns of the investable index proxies (S&P 500, MSCI World).

    Columns are renamed to the labels in :data:`goldman.config.INDEX_PROXIES`.
    """
    csv_path = Path(path) if path is not None else config.data_path(config.INDEX_PROXIES_FILENAME)
    prices = _read_price_csv(csv_path)
    returns = prices.pct_change().dropna(how="all")
    return returns.rename(columns=config.INDEX_PROXIES)
