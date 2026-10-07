# goldman - FE5108 project

**From the Mutual-Fund Theorem to the Security Market Line: does the theory survive real data?**

A reproducible research pipeline for the project, organized by analysis:
mean–variance portfolio choice and estimation error, the CAPM and the empirical
security market line, factor models, and the investment-committee memo. The
portfolio-choice (Stage 1) and CAPM (Stage 2) analyses are implemented and
tested for **both** the US and Singapore universes; factor models and the memo
slot into `src/goldman/analysis/` alongside them.

Analyses are universe-agnostic: a market is described once by a
`goldman.config.Universe` (prices, value weights, currency, risk-free source,
market proxy, factors), so the same Stage 2/3 code runs on either market and
swapping the market proxy is a single argument.

## Project layout

```
.
├── pyproject.toml              # installable package, deps, ruff/pytest config
├── src/goldman/                # the package (src layout)
│   ├── config.py               # shared paths + the Universe registry
│   ├── data/                   # pre-cleaning phase
│   │   ├── prices.py           # price loading, return panels, cleaning
│   │   ├── factors.py          # FF3 factors and the risk-free series
│   │   ├── rates.py            # universe-aware risk-free rates
│   │   ├── markets.py          # market proxies (value-weighted / FF market)
│   │   ├── market_caps.py      # point-in-time value weights (price x shares)
│   │   └── fetch.py            # explicit, network-only fetchers (not imported)
│   ├── analysis/               # portfolio construction and exhibits
│   │   ├── portfolio.py        # tangency weights, 1/N, subsample re-estimation
│   │   ├── portfolio_choice.py # summary stats, weight chart, subsample drift
│   │   └── capm.py             # SCL regressions, empirical SML, GRS test
│   └── __init__.py             # public API
├── notebooks/
│   ├── portfolio_analysis.ipynb   # Stage 1 end to end (single entry point)
│   └── capm_analysis.ipynb        # Stage 2 end to end, US + SG, proxy swap
├── tests/                      # pytest suite for the math and loaders
├── data/                       # raw cached CSVs (committed)
│   └── raw/<universe>/        # per-ticker prices + shares cache
└── outputs/                    # generated exhibits (git-ignored)
```

## Quickstart

```bash
uv sync --extra notebook    # env + Jupyter
uv run jupyter lab notebooks/portfolio_analysis.ipynb   # run all cells
uv run pytest               # 40 tests, coverage gate (min 80%)
uv run ruff check .
```

Each notebook is a reproducible entry point, run top to bottom: it drives the
package, prints its exhibits and writes them to `outputs/`. In
`portfolio_analysis.ipynb`, swap `UNIVERSE` at the top to `"sg"`, and pass
`comparison=None` to `tangency_vs_comparison` / `run_portfolio_analysis` for the
1/N comparison instead of value weights. In `capm_analysis.ipynb`, edit the
`UNIVERSES` list and the `method` loop to cover either market and either proxy.

Network fetchers need the optional `data` extra and open internet; they run
only when invoked — nothing imports them:

```bash
uv sync --extra data
uv run python -m goldman.data.fetch prices       # prices + FF3 factors
uv run python -m goldman.data.fetch caps         # fetch + build value weights
uv run python -m goldman.data.fetch all          # everything
```

Market caps run in two stages so the network is hit once per ticker: `caps-fetch`
pulls prices + shares outstanding into `data/raw/<universe>/`, and
`caps-build` (offline, repeatable) turns that cache into `data/value_weights_*.csv`.
`--refresh` re-pulls; `--universe us` limits a stage to one universe.

`goldman` resolves the repo root from the installed package, so `data/` and
`outputs/` land in the checkout even when invoked from elsewhere. Override with
the `GOLDMAN_ROOT` environment variable if needed.

## Analyses

| Module | What it computes | Outputs |
|---|---|---|
| `analysis/portfolio_choice.py` | Annualised summary statistics; the tangency portfolio `w ∝ Σ⁻¹μ_e`; comparison (value/1/N) weights; full- vs subsample re-estimation | `summary_stats.csv`, `tangency_weights.png`, `value_weights.csv`, `subsample_weights.csv` |
| `analysis/capm.py` | Per-asset `α̂, β̂, t(α̂), R²`; the empirical vs theoretical SML; the GRS joint alpha test | `capm_<universe>_<proxy>_scl.csv`, `..._sml.png`, `..._summary.txt` |
| `data/markets.py` | Market proxy `R_M - R_f`: cap-weighted composite or FF `Mkt-RF` | — |

Adding an analysis means adding a module under `src/goldman/analysis/` and a
section in `notebooks/portfolio_analysis.ipynb`; the notebook reproduces every
stage top to bottom.

## Data

| File | What it is |
|---|---|
| `data/prices_us_monthly.csv` | 30 US names (21 mega-caps + 9 SPDR sector ETFs) |
| `data/prices_sg_monthly.csv` | ~30 large SGX names (approximately the STI) |
| `data/ff3_monthly.csv` | Fama–French 3 factors + RF, decimals, monthly (US) |
| `data/rf_sg_monthly.csv` | Monthly SG risk-free from the MAS compounded SORA Index (month-over-month change), decimal. Raw export and method in `data/raw/mas/` and `rf_sg_monthly.source.txt` |
| `data/raw/<universe>/<ticker>.csv` | Raw per-ticker `close`, `adj_close`, `shares` cache |
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
- **Risk-free rates differ by market.** US uses the Fama–French `RF` column;
  SG uses `data/rf_sg_monthly.csv`, the month-over-month change in the MAS
  compounded SORA Index (the exact daily-compounded return; see
  `rf_sg_monthly.source.txt`). If that file is absent the loader warns and falls
  back to a documented constant `0.003`/month — never the US rate.
- Factor data is US-only: Ken French supplies the US `Mkt-RF`/`SMB`/`HML`, so the
  `ff_market` proxy and Stage 3 factors are unavailable for SG until self-built.
- Residual data limitations: Yahoo has no early share history for a handful of
  names (META before 2022-07; a few SG listings), so those months carry the
  earliest known count back to the sample start — an approximation, but a
  point-in-time price times a plausible share count rather than today's cap.
  The weight chart uses value weights by default; set `comparison=None` in the
  notebook to use the 1/N comparison instead.

## Reproducibility

`notebooks/portfolio_analysis.ipynb` regenerates every Stage 1 exhibit from the
committed raw CSVs when run top to bottom. Generated exhibits go to `outputs/`
(git-ignored). Package versions are pinned in `pyproject.toml` / `uv.lock`.