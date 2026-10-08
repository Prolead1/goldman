"""Central configuration: paths, defaults, and the universe registry.

Everything that used to be hard-coded at the top of the original starter script
lives here so that scripts, the notebook, and the CLI all read the same
settings.
"""

from __future__ import annotations

import os
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
# ``None`` means the file is resolved lazily via ``data_path`` so that the
# registry stays valid even if PROJECT_ROOT moves.
UNIVERSES: dict[str, str] = {
    "synthetic": "sample_synthetic_prices.csv",
    "us": "prices_us_monthly.csv",
    "sg": "prices_sg_monthly.csv",
}

# Ken French data library files (US, monthly, decimals), written by
# ``python -m goldman.data.fetch factors``.
FF3_FILENAME = "ff3_monthly.csv"
FF5_FILENAME = "ff5_monthly.csv"
MOMENTUM_FILENAME = "momentum_monthly.csv"

# Sorted test portfolios (value-weighted) for the CAPM and factor stages.
TEST_PORTFOLIOS: dict[str, str] = {
    "ff25": "ff25_size_bm_monthly.csv",  # 5x5 size x book-to-market
    "mom10": "ff10_momentum_monthly.csv",  # deciles on prior 12-2 month return
}

# Stage 4: a second market proxy for Roll's critique — the Ken French
# developed-markets factors (USD, US T-bill RF), an MSCI World analogue.
DEVELOPED_FILENAME = "developed_3_factors_monthly.csv"

# Stage 4: investable index proxies (Yahoo adjusted closes of the ETFs, so
# total returns net of the funds' small expense ratios): the "local index
# versus MSCI World" pair the handout names.
INDEX_PROXIES_FILENAME = "index_proxies_monthly.csv"
INDEX_PROXIES: dict[str, str] = {
    "SPY": "S&P 500 (SPY ETF)",
    "URTH": "MSCI World (URTH ETF)",
}

# Stage 4 long-history robustness: 10 value-weighted industry portfolios since
# July 1963 (no survivorship: CRSP-wide), with the inputs for their market caps
# (number of firms x average firm size, $m).  The FF3 file starts in 1926 so the
# historical equity premium known at each date can serve as the CAPM prior.
LONG_HISTORY_START = "1963-07-01"
LONG_FACTORS_START = "1926-07-01"
LONG_HISTORY: dict[str, str] = {
    "returns": "long_ind10_vw_monthly.csv",
    "n_firms": "long_ind10_nfirms_monthly.csv",
    "avg_size": "long_ind10_avgsize_monthly.csv",
    "ff3": "long_ff3_monthly.csv",
    "momentum": "long_momentum_monthly.csv",
}

# Value-weight files written by goldman.data.fetch, one per named universe.
# Used as the default comparison portfolio where present.
VALUE_WEIGHT_FILES: dict[str, str] = {
    "us": "value_weights_us.csv",
    "sg": "value_weights_sg.csv",
}


def data_path(filename: str) -> Path:
    """Return an absolute path inside the data directory."""
    return DATA_DIR / filename


def universe_path(universe: str) -> Path:
    """Return the price CSV for a named universe, or raise ``KeyError``."""
    try:
        filename = UNIVERSES[universe]
    except KeyError as exc:
        valid = ", ".join(sorted(UNIVERSES))
        raise KeyError(f"Unknown universe {universe!r}; choose one of: {valid}") from exc
    return data_path(filename)


def value_weights_path(universe: str) -> Path | None:
    """Return the value-weight CSV for a universe, or ``None`` if unregistered."""
    filename = VALUE_WEIGHT_FILES.get(universe)
    return data_path(filename) if filename else None


def raw_universe_dir(universe: str) -> Path:
    """Return the raw per-ticker fetch cache directory for a universe."""
    return RAW_DIR / universe


def ensure_outputs_dir(output_dir: Path | None = None) -> Path:
    """Create and return the output directory (defaults to ``outputs/``)."""
    target = Path(output_dir) if output_dir is not None else OUTPUTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target