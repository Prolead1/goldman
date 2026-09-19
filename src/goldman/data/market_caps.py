"""Value weights for the comparison portfolio.

Loads the market-cap snapshots written by ``scripts/fetch_market_caps.py`` and
turns them into a normalised value-weight series aligned to a return panel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config

__all__ = [
    "load_value_weights",
    "value_weights_from_market_caps",
]


def value_weights_from_market_caps(market_caps: pd.Series) -> pd.Series:
    """Normalise market caps into portfolio weights summing to one."""
    caps = market_caps.astype(float)
    if caps.isna().any():
        missing = ", ".join(caps[caps.isna()].index)
        raise ValueError(f"Market caps missing for: {missing}")
    total = caps.sum()
    if total <= 0:
        raise ValueError("Market caps must sum to a positive number.")
    return caps / total


def load_value_weights(
    path: str | Path | None = None,
    universe: str | None = None,
    assets: pd.Index | list[str] | None = None,
    as_percent: bool = False,
) -> pd.Series:
    """Load value weights, aligning (and renormalising) them to ``assets``.

    The CSV must have an ``asset`` column and either ``weight`` or
    ``market_cap``.  Weights are normalised to sum to one.  When ``assets`` is
    given, the result is reindexed to that order and missing names raise.
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
    if "asset" not in frame.columns:
        raise ValueError(f"{csv_path} must have an 'asset' column.")
    if "market_cap" in frame.columns:
        weights = value_weights_from_market_caps(frame.set_index("asset")["market_cap"])
    elif "weight" in frame.columns:
        weights = frame.set_index("asset")["weight"].astype(float)
        if as_percent:
            weights = weights / 100.0
        weights = weights / weights.sum()
    else:
        raise ValueError(f"{csv_path} must have a 'weight' or 'market_cap' column.")

    weights.name = "value"
    if assets is None:
        return weights
    aligned = weights.reindex(assets)
    if aligned.isna().any():
        missing = ", ".join(aligned[aligned.isna()].index)
        raise ValueError(f"Value weights missing for: {missing}")
    return aligned / aligned.sum()