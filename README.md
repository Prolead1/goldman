# FE 5108 Project

**From the Mutual-Fund Theorem to the Security Market Line: does the theory survive real data?**

A reproducible research pipeline for the project, organized by analysis:
mean–variance portfolio choice and estimation error, the CAPM and the empirical
security market line, factor models, and the investment-committee memo. The
portfolio-choice analysis is implemented and tested; the remaining analyses
slot into `src/goldman/analysis/` alongside it.

## Project layout

```
.
├── run_all.py                  # one entry point: reproduces every exhibit
├── pyproject.toml              # installable package, deps, ruff/pytest config
├── src/goldman/                # the package (src layout)
│   ├── config.py               # shared paths, RF default, universe registry
│   ├── data/                   # pre-cleaning phase
│   │   ├── prices.py           # price loading, return panels, cleaning
│   │   └── factors.py          # FF3 factors and the risk-free series
│   ├── analysis/               # portfolio construction and exhibits
│   │   ├── portfolio.py        # tangency weights, 1/N, subsample re-estimation
│   │   └── portfolio_choice.py # summary stats, weight chart, subsample drift
│   └── cli.py                  # `goldman` command-line interface
├── scripts/download_data.py    # one-shot network downloader (Yahoo/Stooq/FF3)
├── notebooks/portfolio_analysis.ipynb   # thin wrapper over the package
├── tests/                      # pytest suite for the math and loaders
├── data/                       # raw cached CSVs (committed)
└── outputs/                    # generated exhibits (git-ignored)
```

## Quickstart

```bash
uv sync                     # create the env and install goldman (editable)
uv run run_all.py           # default: US universe, all implemented analyses
uv run pytest               # 12 tests
uv run ruff check .
```

Other entry points:

```bash
# CLI (installed console script)
uv run goldman portfolios --universe sg --use-ff3-rf --min-obs 60
uv run python -m goldman portfolios --universe synthetic

# Value weights as the comparison in the weight chart (two-column asset,weight CSV)
uv run goldman portfolios --universe us --value-weights value_weights.csv

# Download the raw data; needs the optional `data` extra and open internet
uv sync --extra data
uv run python scripts/download_data.py
```

`goldman` resolves the repo root from the installed package, so `data/` and
`outputs/` land in the checkout even when invoked from elsewhere. Override with
the `GOLDMAN_ROOT` environment variable if needed.

## Analyses

| Module | What it computes | Outputs |
|---|---|---|
| `analysis/portfolio_choice.py` | Annualised summary statistics; the tangency portfolio `w ∝ Σ⁻¹μ_e`; comparison (value/1/N) weights; full- vs subsample re-estimation | `summary_stats.csv`, `tangency_weights.png`, `subsample_weights.csv` |

Adding an analysis means adding a module here plus a matching subcommand in
`cli.py`; `run_all.py` then reproduces it with the rest.

## Data

| File | What it is |
|---|---|
| `data/prices_us_monthly.csv` | 30 US names (21 mega-caps + 9 SPDR sector ETFs) |
| `data/prices_sg_monthly.csv` | ~30 large SGX names (approximately the STI) |
| `data/ff3_monthly.csv` | Fama–French 3 factors + RF, decimals, monthly |
| `data/sample_synthetic_prices.csv` | **Fake** data for smoke-testing; not for the report |
| `data/coverage_report.txt` | What the downloader fetched |

Cleaning decisions (sparse-ticker drops, fully-observed months) are made once,
in `src/goldman/data/`.

### Honest caveats

- Prices are Yahoo **adjusted closes** (splits + dividends) — a total-return
  proxy. Say so in the report.
- The SG list is *approximately* the STI; verify current constituents before
  using value weights.
- Both universes are **today's members backfilled ten years** — the
  survivorship bias the handout asks you to acknowledge and discuss.
- The weight chart compares tangency weights against **1/N** by default; the
  handout asks for **value weights**. Supply market caps via the two-column CSV
  and pass `--value-weights` (or a `Series` to `run_portfolio_analysis`).

## Reproducibility

`run_all.py` regenerates every exhibit from the committed raw CSVs in one
command. Generated exhibits go to `outputs/` (git-ignored). Package versions
are pinned in `pyproject.toml` / `uv.lock`.