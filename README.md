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
│   │   ├── factors.py          # FF3 factors and the risk-free series
│   │   ├── market_caps.py      # point-in-time value weights (price x shares)
│   │   └── fetch.py            # explicit, network-only fetchers (not imported)
│   ├── analysis/               # portfolio construction and exhibits
│   │   ├── portfolio.py        # tangency weights, 1/N, subsample re-estimation
│   │   └── portfolio_choice.py # summary stats, weight chart, subsample drift
│   └── cli.py                  # `goldman` command-line interface
├── notebooks/portfolio_analysis.ipynb   # thin wrapper over the package
├── tests/                      # pytest suite for the math and loaders
├── data/                       # raw cached CSVs (committed)
│   └── raw/market_caps/        # per-ticker prices + shares cache
└── outputs/                    # generated exhibits (git-ignored)
```

## Quickstart

```bash
uv sync                     # create the env and install goldman (editable)
uv run run_all.py           # default: US universe, all implemented analyses
uv run pytest               # 24 tests
uv run ruff check .
```

Other entry points:

```bash
# CLI (installed console script)
uv run goldman portfolios --universe sg --use-ff3-rf --min-obs 60
uv run python -m goldman portfolios --universe synthetic

# Force the 1/N comparison instead of the default value weights
uv run goldman portfolios --universe us --equal-weights

# Override value weights with your own CSV (asset, weight|market_cap)
uv run goldman portfolios --universe us --value-weights my_weights.csv

# Network fetchers; need the optional `data` extra and open internet.
# goldman.data.fetch runs only when invoked — nothing imports it.
uv sync --extra data
uv run python -m goldman.data.fetch prices       # prices + FF3 factors
uv run python -m goldman.data.fetch caps         # fetch + build value weights
uv run python -m goldman.data.fetch all          # everything
```

Market caps run in two stages so the network is hit once per ticker: `caps-fetch`
pulls prices + shares outstanding into `data/raw/market_caps/<universe>/`, and
`caps-build` (offline, repeatable) turns that cache into `data/value_weights_*.csv`.
`--refresh` re-pulls; `--universe us` limits a stage to one universe.

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
| `data/raw/market_caps/<universe>/<ticker>.csv` | Raw per-ticker `close`, `adj_close`, `shares` cache |
| `data/value_weights_us.csv` | US point-in-time market cap -> monthly value weights |
| `data/value_weights_sg.csv` | SG point-in-time market cap -> monthly value weights |
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
- Value weights are **point-in-time**: `market cap(t) = split-unadjusted
  close(t) x shares outstanding(t)`, with shares forward-filled from their
  report dates, so no month uses a future share count. The weight CSV is long
  format (`asset, date, market_cap, weight`); the chart uses the sample-average
  vector and the full monthly panel is written to `outputs/value_weights.csv`.
- Residual data limitations: Yahoo has no early share history for a handful of
  names (META before 2022-07; a few SG listings), so those months carry the
  earliest known count back to the sample start — an approximation, but a
  point-in-time price times a plausible share count rather than today's cap.
  The weight chart uses value weights by default and falls back to 1/N where no
  weights exist (e.g. the synthetic sample); pass `--equal-weights` to force 1/N.

## Reproducibility

`run_all.py` regenerates every exhibit from the committed raw CSVs in one
command. Generated exhibits go to `outputs/` (git-ignored). Package versions
are pinned in `pyproject.toml` / `uv.lock`.