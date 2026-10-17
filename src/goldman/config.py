"""Central configuration: paths, defaults, and the universe registry.

Everything that used to be hard-coded at the top of the original starter script
lives here so that scripts, the notebook, and the CLI all read the same
settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------- paths
PACKAGE_ROOT: Path = Path(__file__).resolve().parent


def _resolve_project_root() -> Path:
    """Locate the repository root.

    Honours ``GOLDMAN_ROOT`` if set, otherwise walks up from the installed
    package to the checkout that contains ``data/``.  Falls back to the
    current working directory when the package is installed elsewhere.
    """
    env_root = os.environ.get("GOLDMAN_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    candidate = PACKAGE_ROOT.parents[1]  # src/goldman -> src -> repo root
    if (candidate / "data").is_dir():
        return candidate
    return Path.cwd()


PROJECT_ROOT: Path = _resolve_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Raw per-ticker fetch cache written by goldman.data.fetch before any
# conversion: data/raw/<universe>/<ticker>.csv
RAW_DIR: Path = DATA_DIR / "raw"

# ------------------------------------------------------------------ defaults
RF_MONTHLY: float = 0.003  # fallback monthly risk-free rate (use FF3/MAS)
MIN_OBS: int = 90  # drop assets with fewer clean monthly observations
TRADING_MONTHS: int = 12  # annualisation factor for monthly data

# --------------------------------------------------------- universe registry


@dataclass(frozen=True)
class Universe:
    """One investable market and everything that differs across markets.

    Writing an analysis once and parameterising it by :class:`Universe` is what
    lets the same Stage 2/3 code run on both the US and Singapore universes.

    Attributes
    ----------
    prices_file:
        Committed monthly price CSV under ``data/``.
    value_weights_file:
        Committed long-format value-weight CSV, or ``None`` when unavailable.
    currency:
        Currency the returns are denominated in (reporting only).
    rf_source:
        Where the risk-free rate comes from:
        ``"ff3"`` uses the ``RF`` column of the bundled Fama–French file;
        ``"file"`` reads ``rf_file`` (a committed monthly series);
        ``"constant"`` returns ``rf_constant``.
    rf_file:
        Filename of an optional committed monthly RF series (source ``"file"``).
    rf_constant:
        Documented fallback monthly rate used when no series is available.
    market_proxy:
        Default market excess-return construction: ``"value_weighted"``
        (cap-weighted composite of the universe) or ``"ff_market"``
        (the Fama–French ``Mkt-RF`` factor).
    factors:
        Factor columns available for Stage 3 in ``ff3_monthly.csv``.
    """

    name: str
    prices_file: str
    value_weights_file: str | None = None
    currency: str = "USD"
    rf_source: str = "constant"
    rf_file: str | None = None
    rf_constant: float = RF_MONTHLY
    market_proxy: str = "value_weighted"
    factors: tuple[str, ...] = ()


# ``None`` filenames are resolved lazily via ``data_path`` so the registry stays
# valid even if PROJECT_ROOT moves.
UNIVERSES: dict[str, Universe] = {
    "synthetic": Universe(
        name="synthetic",
        prices_file="sample_synthetic_prices.csv",
        currency="USD",
        rf_source="constant",
    ),
    "us": Universe(
        name="us",
        prices_file="prices_us_monthly.csv",
        value_weights_file="value_weights_us.csv",
        currency="USD",
        rf_source="ff3",
        market_proxy="value_weighted",
        factors=("Mkt-RF", "SMB", "HML"),
    ),
    "sg": Universe(
        name="sg",
        prices_file="prices_sg_monthly.csv",
        value_weights_file="value_weights_sg.csv",
        currency="SGD",
        rf_source="file",
        rf_file="rf_sg_monthly.csv",
        rf_constant=RF_MONTHLY,
        market_proxy="value_weighted",
        factors=(),
    ),
}

FF3_FILENAME = "ff3_monthly.csv"

# Long US history for the industry momentum factor (report, after the memo's
# factor-hunting section): Ken French's 10 value-weighted industry portfolios,
# which cover every CRSP stock and so carry no survivorship bias, plus the
# market, size, value and momentum factors to test it against.  The industries
# start at 1963-07, where the Fama and French (1993) sample starts; the factors
# go back to 1926.
LONG_HISTORY_START = "1963-07-01"
LONG_FACTORS_START = "1926-07-01"
LONG_HISTORY: dict[str, str] = {
    "industries": "long_ind10_vw_monthly.csv",
    "ff3": "long_ff3_monthly.csv",
    "momentum": "long_momentum_monthly.csv",
}

# Backwards-compatible filename map; the registry above is the source of truth.
VALUE_WEIGHT_FILES: dict[str, str] = {
    u.name: u.value_weights_file for u in UNIVERSES.values() if u.value_weights_file
}


def get_universe(universe: str) -> Universe:
    """Return the :class:`Universe` registry entry, or raise ``KeyError``."""
    try:
        return UNIVERSES[universe]
    except KeyError as exc:
        valid = ", ".join(sorted(UNIVERSES))
        raise KeyError(f"Unknown universe {universe!r}; choose one of: {valid}") from exc


def universe_names() -> list[str]:
    """Sorted names of every registered universe."""
    return sorted(UNIVERSES)


def data_path(filename: str) -> Path:
    """Return an absolute path inside the data directory."""
    return DATA_DIR / filename


def universe_path(universe: str) -> Path:
    """Return the price CSV for a named universe, or raise ``KeyError``."""
    return data_path(get_universe(universe).prices_file)


def value_weights_path(universe: str) -> Path | None:
    """Return the value-weight CSV for a universe, or ``None`` if unregistered."""
    filename = get_universe(universe).value_weights_file
    return data_path(filename) if filename else None


def raw_universe_dir(universe: str) -> Path:
    """Return the raw per-ticker fetch cache directory for a universe."""
    return RAW_DIR / universe


def ensure_outputs_dir(output_dir: Path | None = None) -> Path:
    """Create and return the output directory (defaults to ``outputs/``)."""
    target = Path(output_dir) if output_dir is not None else OUTPUTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target