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

FF3_FILENAME = "ff3_monthly.csv"

# Value-weight files written by scripts/fetch_market_caps.py, one per named
# universe.  Used as the default comparison portfolio where present.
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


def ensure_outputs_dir(output_dir: Path | None = None) -> Path:
    """Create and return the output directory (defaults to ``outputs/``)."""
    target = Path(output_dir) if output_dir is not None else OUTPUTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target