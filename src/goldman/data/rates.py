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
    "monthly_rf_from_sora_export",
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


def monthly_rf_from_sora_export(
    path: str | Path,
    *,
    complete_months_only: bool = True,
) -> pd.DataFrame:
    """Build a monthly SG risk-free return from a MAS SORA export.

    The MAS "Domestic Interest Rates" download carries daily SORA quotes and the
    compounded **SORA Index**.  The exact one-month risk-free return is the
    month-over-month change in that index,

        ``RF_t = SORA_Index(month_end_t) / SORA_Index(month_end_{t-1}) - 1``,

    which equals compounding the daily overnight rate at actual/365 (verified to
    ~1e-11 against the daily quotes).  Averaging the annualised daily quotes and
    dividing by 12 is only a first-order approximation and is not used here.

    Returns a frame with columns ``Date`` (``YYYY-MM``) and ``RF``.  With
    ``complete_months_only`` the final partial month of a mid-month download is
    dropped.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = next((i for i, line in enumerate(lines) if "SORA Value Date" in line), None)
    if header is None:
        raise ValueError(f"{path} does not look like a MAS SORA export.")

    frame = pd.read_csv(path, skiprows=header, header=0, engine="python")
    frame = frame.rename(
        columns={"SORA Value Date": "year", "Unnamed: 1": "month", "Unnamed: 2": "day"}
    )
    frame["year"] = frame["year"].ffill()
    frame["month"] = frame["month"].ffill()
    frame["sora"] = pd.to_numeric(frame["SORA"], errors="coerce")
    frame["sora_index"] = pd.to_numeric(frame["SORA Index"], errors="coerce")
    frame = frame[frame["sora"].notna() & frame["year"].astype(str).str.fullmatch(r"\d{4}")].copy()
    dates = pd.to_datetime(
        frame["year"].astype(int).astype(str)
        + " "
        + frame["month"].str.strip()
        + " "
        + frame["day"].astype(int).astype(str),
        format="%Y %b %d",
    )
    frame = frame.assign(date=dates).sort_values("date").set_index("date")

    month_end_index = frame["sora_index"].resample("ME").last()
    monthly = month_end_index.pct_change()

    if complete_months_only:
        last_obs = frame.index.to_series().resample("ME").max()
        complete = last_obs.reindex(monthly.index) >= (monthly.index - pd.Timedelta(days=3))
        monthly = monthly[complete]

    out = monthly.dropna().rename("RF").rename_axis("Date").reset_index()
    out["Date"] = out["Date"].dt.strftime("%Y-%m")
    return out[out["RF"].notna()]


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
