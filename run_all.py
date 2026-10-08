"""Regenerate every exhibit with one command.

Executes ``notebooks/portfolio_analysis.ipynb`` top to bottom in a fresh kernel
and saves it with its outputs, so the notebook stays the single source of
truth; every exhibit lands in ``outputs/``.  No network access is needed.

    uv sync --extra notebook
    uv run python run_all.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "portfolio_analysis.ipynb"


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK)
    print(f"executed {NOTEBOOK.relative_to(ROOT)}; exhibits written to outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
