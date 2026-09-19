"""Factor and risk-free-rate loading.

The Ken French file supplies the market, size and value factors plus the
monthly risk-free rate.  ``monthly_rf`` aligns that rate to an arbitrary
return index and falls back to a constant outside the factor file's coverage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config

__all__ = [
    "load_ff3",
    "monthly_rf",
    "to_period_index",
]


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