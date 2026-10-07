"""Universe-aware risk-free rates.

The US universe uses the Fama–French ``RF`` column.  Markets without a bundled
factor file (Singapore) read an optional committed monthly series
``data/<rf_file>``; when that file is absent the loader falls back to the
universe's documented constant and warns — rather than silently borrowing
another market's rate, which is what a bare FF3 lookup would do for SG dates
because the calendars overlap.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from .. import config
from .factors import load_ff3, monthly_rf

__all__ = [
    "load_rf_file",
    "risk_free_series",
]


def load_rf_file(path: str | Path) -> pd.Series:
    """Read a committed monthly risk-free series.

    Accepts a ``Date`` column (``YYYY-MM``) plus an ``RF`` column of monthly
    decimals, or the same shape written with the date as the index.
    """
    frame = pd.read_csv(path)
    if "Date" in frame.columns:
        frame["Date"] = pd.PeriodIndex(pd.to_datetime(frame["Date"]), freq="M")
        frame = frame.set_index("Date")
    else:
        frame.index = pd.PeriodIndex(pd.to_datetime(frame.index), freq="M")
        frame.index.name = "Date"
    if "RF" not in frame.columns:
        raise ValueError(f"{path} must have an 'RF' column.")
    return frame["RF"].astype(float).sort_index().rename("RF")


def risk_free_series(
    universe: str,
    index: pd.Index,
    ff3: pd.DataFrame | None = None,
) -> pd.Series:
    """Monthly risk-free rate for ``universe``, aligned to ``index``.

    Dispatch follows the universe's ``rf_source``:

    * ``"ff3"`` — the Fama–French ``RF`` column (loaded on demand), falling back
      to the universe constant outside its coverage.
    * ``"file"`` — the committed ``rf_file`` if present, forward-filled onto the
      calendar, otherwise a warning and the universe constant.
    * ``"constant"`` — the universe constant.
    """
    spec = config.get_universe(universe)

    if spec.rf_source == "ff3":
        if ff3 is None and config.data_path(config.FF3_FILENAME).exists():
            ff3 = load_ff3()
        return monthly_rf(index, ff3=ff3, fallback=spec.rf_constant)

    if spec.rf_source == "file":
        path = config.data_path(spec.rf_file) if spec.rf_file else None
        if path is not None and path.exists():
            periods = pd.PeriodIndex(pd.to_datetime(index), freq="M")
            series = load_rf_file(path).reindex(periods).ffill()
            series.index = index
            return series.fillna(spec.rf_constant).rename("RF")

    if spec.rf_source not in {"constant", "file"}:
        raise ValueError(f"Unknown rf_source {spec.rf_source!r} for universe {universe!r}.")

    if spec.rf_source == "file":
        warnings.warn(
            f"No risk-free series at {config.data_path(spec.rf_file)} for universe "
            f"{universe!r}; falling back to a constant {spec.rf_constant:.4f}/month. "
            "Drop a real MAS/SORA series at that path to use it.",
            stacklevel=2,
        )
    return pd.Series(spec.rf_constant, index=index, name="RF")
