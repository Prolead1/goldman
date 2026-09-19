#!/usr/bin/env python
"""Single entry point that reproduces every exhibit from raw cached data.

The report's reproducibility requirement asks for one command that regenerates
all exhibits from the raw CSVs in ``data/``.  This script is that command::

    uv run run_all.py                 # default: US universe, all analyses
    uv run run_all.py --universe sg
    uv run run_all.py --universe synthetic --output-dir outputs/smoke

Analyses are added here as they are implemented; ``goldman.cli`` owns the logic.
"""

from __future__ import annotations

import sys

from goldman.cli import main

DEFAULT_ARGS = ["all", "--universe", "us"]

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or DEFAULT_ARGS))