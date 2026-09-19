"""Point-in-time value weights for the comparison portfolio.

Market cap at date ``t`` is ``close(t) x shares_outstanding(t)``.  Shares are
forward-filled from their report dates so every month uses only information
that was available then — the value-weighted benchmark therefore carries no
look-ahead bias, unlike a single "today's market cap" snapshot applied to a
ten-year panel.

The fetch step writes a raw per-ticker cache
(``data/raw/market_caps/<universe>/<ticker>.csv``) that holds everything pulled
from the network in one pass; :func:`build_value_weights_from_raw` converts that
cache into a long-format ``asset, date, market_cap, weight`` CSV and
:func:`load_value_weights` turns it into a date-by-asset weight panel aligned to
the return panel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config

__all__ = [
    "market_caps_from_shares",
    "unadjust_close",
    "value_weights_from_market_caps",
    "value_weights_from_raw",
    "build_value_weights_from_raw",
    "load_value_weights",
]


def unadjust_close(close: pd.Series, splits: pd.Series) -> pd.Series:
    """Undo Yahoo's split back-adjustment so price x shares is a true market cap.

    Yahoo's ``Close`` (even with ``auto_adjust=False``) is retroactively divided
    by every later split, while ``get_shares_full`` reports the share count as it
    was *at the time*.  Multiplying the two understates pre-split caps, so we
    scale each pre-split month back up by the split ratio.  Monthly bars are
    month-end values, so the split month's bar is already post-split and only
    earlier months are rescaled.  ``splits`` is indexed like ``close`` with the
    split ratio on the split month and zero elsewhere.
    """
    raw = close.astype(float).copy()
    month_end = raw.index + pd.offsets.MonthEnd(0)
    for split_date, ratio in splits[splits.fillna(0) > 0].items():
        raw.loc[month_end < split_date] *= float(ratio)
    return raw


def market_caps_from_shares(
    prices: pd.DataFrame,
    shares: pd.DataFrame,
) -> pd.DataFrame:
    """Point-in-time market cap = price x shares outstanding, aligned by date.

    Both frames are indexed by date and share the same asset columns.  Shares
    are forward-filled onto the price index so a month is weighted with the most
    recently *reported* share count — never a future one.
    """
    prices = prices.sort_index()
    shares = shares.sort_index()
    aligned = shares.reindex(shares.index.union(prices.index)).ffill().reindex(prices.index)
    caps = prices.astype(float) * aligned.astype(float)
    if caps.isna().any().any():
        missing = caps.columns[caps.isna().any()].tolist()
        raise ValueError(f"Market caps missing for: {', '.join(missing)}")
    return caps


def value_weights_from_market_caps(
    market_caps: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Normalise market caps into portfolio weights.

    A ``Series`` is treated as one cross-section and returns a ``Series``.  A
    ``DataFrame`` is treated as a date-by-asset panel and each row is normalised
    independently, returning a ``DataFrame``.
    """
    caps = market_caps.astype(float)
    if isinstance(caps, pd.Series):
        if caps.isna().any():
            missing = ", ".join(caps[caps.isna()].index)
            raise ValueError(f"Market caps missing for: {missing}")
        total = caps.sum()
        if total <= 0:
            raise ValueError("Market caps must sum to a positive number.")
        return caps / total

    if caps.isna().any().any():
        totals = caps.sum(axis=1, skipna=True)
        if (totals <= 0).any():
            bad = caps.index[totals <= 0]
            raise ValueError(f"Market caps missing for dates: {', '.join(map(str, bad))}")
        # Some assets are listed after the panel starts; weight each date over
        # the names that actually have a cap then.
        return caps.div(totals, axis=0)
    totals = caps.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Market caps must sum to a positive number at every date.")
    return caps.div(totals, axis=0)


def value_weights_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn one raw per-ticker frame (``close``, ``shares``) into a cap series.

    Shares are forward-filled from their report dates (point-in-time).  Where
    Yahoo has no early share history, the earliest known count is carried back
    so the asset is still valued with a point-in-time *price*; this assumes a
    roughly constant share count before the first report and is documented as a
    data-availability approximation.
    """
    if "close" not in raw.columns or "shares" not in raw.columns:
        raise ValueError("Raw market-cap frame must have 'close' and 'shares' columns.")
    shares = raw["shares"].ffill().bfill()
    cap = raw["close"].astype(float) * shares.astype(float)
    cap.name = "market_cap"
    return cap.to_frame()


def build_value_weights_from_raw(
    universe: str,
    assets: pd.Index | list[str] | None = None,
) -> pd.DataFrame:
    """Read the raw per-ticker cache for ``universe`` and build net weights.

    Returns a long-format frame with columns ``asset, date, market_cap,
    weight``; the weight column is normalised within each date.  Tickers whose
    raw file is absent or has no usable cap are omitted.
    """
    raw_dir = config.raw_market_caps_dir(universe)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"No raw market-cap cache at {raw_dir}; run "
            "`python -m goldman.data.fetch caps-fetch` first."
        )

    names = list(assets) if assets is not None else sorted(p.stem for p in raw_dir.glob("*.csv"))
    caps: dict[str, pd.Series] = {}
    for asset in names:
        path = raw_dir / f"{asset}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
        series = value_weights_from_raw(raw)["market_cap"].dropna()
        if not series.empty:
            caps[asset] = series

    if not caps:
        raise ValueError(f"No usable market-cap data in {raw_dir}.")

    cap_panel = pd.DataFrame(caps).sort_index()
    weights = value_weights_from_market_caps(cap_panel)

    def _long(panel: pd.DataFrame, value_name: str) -> pd.DataFrame:
        return (
            panel.rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="asset", value_name=value_name)
            .dropna(subset=[value_name])
        )

    long = _long(cap_panel, "market_cap").merge(
        _long(weights, "weight"), on=["date", "asset"], how="left"
    )
    long = long.sort_values(["date", "asset"]).reset_index(drop=True)
    return long[["asset", "date", "market_cap", "weight"]]


def _weights_from_frame(
    frame: pd.DataFrame,
    csv_path: Path,
    as_percent: bool,
) -> tuple[pd.DataFrame, bool]:
    """Read a long/dated or legacy static weight frame into a date-by-asset panel.

    Returns ``(panel, is_static)``; a static file is a single cross-section that
    the caller broadcasts across the requested dates.
    """
    if "asset" not in frame.columns:
        raise ValueError(f"{csv_path} must have an 'asset' column.")

    if "date" not in frame.columns:
        # Legacy static snapshot: one cross-section, broadcast to `dates` later.
        if "market_cap" in frame.columns:
            row = value_weights_from_market_caps(frame.set_index("asset")["market_cap"])
        elif "weight" in frame.columns:
            row = frame.set_index("asset")["weight"].astype(float)
            if as_percent:
                row = row / 100.0
            row = row / row.sum()
        else:
            raise ValueError(f"{csv_path} must have a 'weight' or 'market_cap' column.")
        return row.to_frame().T, True

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "market_cap" in frame.columns:
        caps = frame.pivot(index="date", columns="asset", values="market_cap").sort_index()
        return value_weights_from_market_caps(caps), False
    if "weight" in frame.columns:
        weights = frame.pivot(index="date", columns="asset", values="weight").astype(float)
        if as_percent:
            weights = weights / 100.0
        return weights.div(weights.sum(axis=1), axis=0), False
    raise ValueError(f"{csv_path} must have a 'weight' or 'market_cap' column.")


def load_value_weights(
    path: str | Path | None = None,
    universe: str | None = None,
    assets: pd.Index | list[str] | None = None,
    dates: pd.Index | list | None = None,
    as_percent: bool = False,
) -> pd.DataFrame:
    """Load value weights as a date-by-asset panel aligned to ``dates``.

    The CSV must have an ``asset`` column.  A dated file adds a ``date`` column
    and may carry ``market_cap`` or ``weight``; a legacy static file (no
    ``date``) is broadcast across ``dates``.  Weights are normalised to sum to
    one within each row.  When ``dates`` is given the panel is forward-filled
    onto that index — a month uses the most recent weight snapshot, never a
    future one — and missing names/periods raise.
    """
    if (path is None) == (universe is None):
        raise ValueError("Provide exactly one of `path` or `universe`.")
    if path is not None:
        csv_path = Path(path)
    else:
        registered = config.value_weights_path(universe)
        if registered is None:
            raise KeyError(f"No value-weight file registered for universe {universe!r}")
        csv_path = registered

    frame = pd.read_csv(csv_path)
    weights, static = _weights_from_frame(frame, csv_path, as_percent)

    if dates is not None:
        target = pd.DatetimeIndex(dates)
        if static:
            weights = pd.DataFrame(
                [weights.iloc[0].to_numpy()], index=target, columns=weights.columns
            )
        else:
            weights = weights.sort_index()
            weights = weights.reindex(weights.index.union(target)).ffill().reindex(target)
    elif static:
        weights = weights.reset_index(drop=True)

    # Align columns only after the date window is applied, so gaps outside the
    # sample (e.g. pre-listing) do not masquerade as missing weights.
    if assets is not None:
        weights = weights.reindex(columns=assets)
        if weights.isna().any().any():
            missing = ", ".join(weights.columns[weights.isna().any()])
            raise ValueError(f"Value weights missing for: {missing}")

    weights = weights.div(weights.sum(axis=1), axis=0)
    weights.index.name = "date"
    weights.columns.name = "asset"
    return weights