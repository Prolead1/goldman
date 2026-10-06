"""Reproduce Kai's three-factor analysis from the cached project data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from goldman.analysis.kai_factors import main  # noqa: E402

if __name__ == "__main__":
    main()
