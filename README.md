# goldman - FE5108 project

**From the Mutual-Fund Theorem to the Security Market Line: does the theory survive real data?**

A reproducible research pipeline for the project, organized by analysis:
mean–variance portfolio choice and estimation error, the CAPM and the empirical
security market line, factor models, and the investment-committee memo. All
four stages are implemented and tested; `report/draft_report.md` holds the draft
executive summary, figure captions and Stage 4 memo for the PDF.

## Project layout

```
.
├── pyproject.toml              # installable package, deps, ruff/pytest config
├── src/goldman/                # the package (src layout)
│   ├── config.py               # shared paths, RF default, universe registry
│   ├── data/                   # pre-cleaning phase
│   │   ├── prices.py           # price loading, return panels, cleaning
│   │   ├── factors.py          # Ken French factors, RF, test portfolios, alignment
│   │   ├── market_caps.py      # point-in-time value weights (price x shares)
│   │   └── fetch.py            # explicit, network-only fetchers (not imported)
│   ├── analysis/               # portfolio construction and exhibits
│   │   ├── portfolio.py        # tangency weights, 1/N, subsample re-estimation
│   │   ├── portfolio_choice.py # Stage 1: summary stats, weight chart, drift
│   │   ├── regression.py       # shared econometrics: OLS + Newey–West, GRS, Fama–MacBeth
│   │   ├── estimators.py       # Ledoit–Wolf covariance, CAPM-prior mean shrinkage
│   │   ├── capm.py             # Stage 2: CAPM table, joint alpha test, empirical SML
│   │   ├── factor_models.py    # Stage 3: FF3 / Carhart / FF5 alphas and the verdict
│   │   ├── self_built.py       # Stage 3: a self-built momentum factor (long–short)
│   │   ├── roll.py             # Stage 4: Roll's critique (CAPM under other proxies)
│   │   └── backtest.py         # Stage 4: out-of-sample horse race of portfolio rules
│   └── __init__.py             # public API
├── notebooks/portfolio_analysis.ipynb   # single entry point: Stages 1–4 end to end
├── run_all.py                  # executes the notebook headlessly (same entry point)
├── report/draft_report.md      # draft executive summary, captions, Stage 4 memo
├── tests/                      # pytest suite for the math and loaders
├── data/                       # raw cached CSVs (committed)
│   └── raw/<universe>/        # per-ticker prices + shares cache
└── outputs/                    # generated exhibits (git-ignored)
```

## Quickstart

```bash
uv sync --extra notebook    # env + Jupyter
uv run python run_all.py    # regenerate every exhibit (executes the notebook)
uv run jupyter lab notebooks/portfolio_analysis.ipynb   # or run it interactively
uv run pytest               # 107 tests, coverage gate (min 80%)
uv run ruff check .
```

The notebook is the single reproducible entry point: it drives the package,
prints every Stage 1–4 exhibit and writes them to `outputs/`. Swap `UNIVERSE` at
the top to `"sg"` for Stage 1 (Stages 2–4 use US Ken French data and assert
`UNIVERSE == "us"`), and pass `comparison=None` to `tangency_vs_comparison` /
`run_portfolio_analysis` for the 1/N comparison instead of value weights.

Network fetchers need the optional `data` extra and open internet; they run
only when invoked — nothing imports them:

```bash
uv sync --extra data
uv run python -m goldman.data.fetch prices       # prices + Ken French data
uv run python -m goldman.data.fetch factors      # Ken French factors + portfolios only
uv run python -m goldman.data.fetch proxies      # S&P 500 + MSCI World ETFs (Stage 4)
uv run python -m goldman.data.fetch caps         # fetch + build value weights
uv run python -m goldman.data.fetch all          # everything
```

Market caps run in two stages so the network is hit once per ticker: `caps-fetch`
pulls prices + shares outstanding into `data/raw/<universe>/`, and
`caps-build` (offline, repeatable) turns that cache into `data/value_weights_*.csv`.
`--refresh` re-pulls; `--universe us` limits a stage to one universe.

`scipy` (a core dependency) supplies only the F, t and binomial distributions
for the tests; every estimator is written out in numpy.

`goldman` resolves the repo root from the installed package, so `data/` and
`outputs/` land in the checkout even when invoked from elsewhere. Override with
the `GOLDMAN_ROOT` environment variable if needed.

## Analyses

| Module | What it computes | Outputs |
|---|---|---|
| `analysis/portfolio_choice.py` | Stage 1. Annualised summary statistics; the tangency portfolio `w ∝ Σ⁻¹μ_e`; comparison (value/1/N) weights; full- vs subsample re-estimation; no-short and shrinkage versions | `summary_stats.csv`, `tangency_weights.png`, `value_weights.csv`, `subsample_weights.csv`, `taming_table.csv`, `taming_weights.png` |
| `analysis/capm.py` | Stage 2. Security characteristic regressions (OLS and Newey–West t-stats); GRS test and the chance count of \|t\| > 2; Fama–MacBeth SML with Shanken-corrected tests of intercept = 0 and slope = market premium; SML robustness across test assets and periods | `capm_table.csv`, `alpha_joint_test.csv`, `sml.png`, `sml_tests.csv`, `sml_robustness.csv` (+ `ff25_` variants) |
| `analysis/factor_models.py` | Stage 3. Factor premia; CAPM → FF3 → Carhart 4 (and FF5) alphas per asset; GRS / mean \|α\| / R² per model; absorbed-or-survives verdict; loadings | `factor_summary.csv`, `alpha_comparison.png/.csv`, `model_summary.csv`, `anomaly_verdict.csv`, `factor_loadings.csv` (+ `ff25_`, `mom10_`, `ff5_` variants) |
| `analysis/self_built.py` | Stage 3 (appendix). Momentum built from returns (t−12..t−2 signal, long–short, monthly), validated against Ken French's Mom; CAPM/FF3 alphas, turnover, net-of-cost and break-even cost | `self_built_momentum.csv` |
| `analysis/roll.py` | Stage 4. Roll's critique: Stage 2 re-run under the CRSP, S&P 500, MSCI World, Ken French developed-markets, universe value-weighted, universe 1/N and ex-post tangency proxies; GRS as a Sharpe-ratio distance | `roll_summary.csv`, `roll_betas.csv/.png`, `roll_alphas.csv/.png` |
| `analysis/backtest.py` | Stage 4. Rolling out-of-sample horse race (value-weighted, 1/N, min variance, plug-in / long-only / shrinkage tangency); Sharpe, Jobson–Korkie–Memmel test, certainty equivalents, drawdown, turnover, net-of-cost Sharpe | `horse_race_us30.csv/.png`, `horse_race_industries_1963.csv/.png`, `two_fund_allocation.csv` |
| `analysis/estimators.py` | Ledoit–Wolf (2004) constant-correlation covariance; CAPM-implied means; James–Stein shrinkage toward them | — |
| `analysis/regression.py` | Shared econometrics: per-asset OLS with Newey–West errors, GRS (1989), chance exceedances, Fama–MacBeth (1973) with the Shanken (1992) correction | — |

Adding an analysis means adding a module under `src/goldman/analysis/` and a
section in `notebooks/portfolio_analysis.ipynb`; the notebook reproduces every
stage top to bottom.

## Data

| File | What it is |
|---|---|
| `data/prices_us_monthly.csv` | 30 US names (21 mega-caps + 9 SPDR sector ETFs) |
| `data/prices_sg_monthly.csv` | ~30 large SGX names (approximately the STI) |
| `data/ff3_monthly.csv` | Fama–French 3 factors + RF (one-month T-bill), decimals, monthly |
| `data/ff5_monthly.csv` | Fama–French 5 factors (2x3) + RF — robustness model |
| `data/momentum_monthly.csv` | Ken French momentum factor `Mom` (prior 12–2 month return) |
| `data/ff25_size_bm_monthly.csv` | 25 value-weighted size x book-to-market portfolios (Stage 2–3 test assets) |
| `data/ff10_momentum_monthly.csv` | 10 value-weighted momentum-decile portfolios (Stage 3 test assets) |
| `data/developed_3_factors_monthly.csv` | Ken French developed-markets factors, USD, US T-bill RF — the second market proxy (MSCI World analogue) |
| `data/long_ind10_vw_monthly.csv` | 10 value-weighted US industry portfolios, 1963-07 – 2026-08 (Stage 4 long-history race) |
| `data/long_ind10_nfirms_monthly.csv`, `data/long_ind10_avgsize_monthly.csv` | Number of firms and average firm size ($m) per industry — their product is the industry market cap |
| `data/long_ff3_monthly.csv` | FF3 + RF since 1926-07 (the historical equity premium used as the CAPM prior and for the two-fund split) |
| `data/long_momentum_monthly.csv` | Ken French momentum factor since 1927 (validates the self-built factor) |
| `data/index_proxies_monthly.csv` | Monthly adjusted closes of SPY (S&P 500) and URTH (MSCI World) — Stage 4 proxies |
| `data/raw/<universe>/<ticker>.csv` | Raw per-ticker `close`, `adj_close`, `shares` cache |
| `data/value_weights_us.csv` | US point-in-time market cap -> monthly value weights |
| `data/value_weights_sg.csv` | SG point-in-time market cap -> monthly value weights |
| `data/sample_synthetic_prices.csv` | **Fake** data for smoke-testing; not for the report |
| `data/coverage_report.txt` | What the price downloader fetched |
| `data/coverage_factors.txt` | What the Ken French downloader fetched (all one library vintage) |

Cleaning decisions (sparse-ticker drops, fully-observed months) are made once,
in `src/goldman/data/`.

### Honest caveats

- Prices are Yahoo **adjusted closes** (splits + dividends) — a total-return
  proxy. Say so in the report.
- The SG list is *approximately* the STI; verify current constituents before
  using value weights.
- Both universes are **today's members backfilled ten years** — the
  survivorship bias the handout asks you to acknowledge and discuss. In Stage 2
  it is a plausible reason the 30-asset SML looks steeper than theory (the
  steepness rests on NVDA alone) while the selection-free Ken French
  portfolios show the usual flat SML — neither deviation is statistically
  significant.
- Stages 2–4 use the US Ken French market proxy (CRSP value-weighted) and
  factors, so they apply to the `us` universe only; an SG run needs a local
  market proxy, an SGD risk-free rate and self-built factors.
- The Stage 4 horse race on our universe has only 59 out-of-sample months and
  inherits the survivorship bias (the universe's value-weighted index beat the
  investable CRSP index by 5.3% p.a.); the 1963–2026 industry race is the
  survivorship-free check.
- Value weights are **point-in-time**: `market cap(t) = split-unadjusted
  close(t) x shares outstanding(t)`, with shares forward-filled from their
  report dates, so no month uses a future share count. The weight CSV is long
  format (`asset, date, market_cap, weight`); the chart uses the sample-average
  vector and the full monthly panel is written to `outputs/value_weights.csv`.
- Residual data limitations: Yahoo has no early share history for a handful of
  names (META before 2022-07; a few SG listings), so those months carry the
  earliest known count back to the sample start — an approximation, but a
  point-in-time price times a plausible share count rather than today's cap.
  The weight chart uses value weights by default; set `comparison=None` in the
  notebook to use the 1/N comparison instead.

## Reproducibility

`notebooks/portfolio_analysis.ipynb` regenerates every Stage 1–4 exhibit from the
committed raw CSVs when run top to bottom, with no network access (under 10
seconds). Generated exhibits go to `outputs/`
(git-ignored). Package versions are pinned in `pyproject.toml` / `uv.lock`; the
exhibits were produced with:

| Package | Version |
|---|---|
| Python | 3.13.11 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| jupyterlab / ipykernel / nbconvert | 4.6.3 / 7.3.0 / 7.17.1 |
| pandas-datareader / yfinance (data refresh only) | 0.11.1 / 1.7.0 |
| pytest / ruff (dev) | 9.1.1 / 0.16.8 |

**macOS note.** If `import goldman` fails with `ModuleNotFoundError` inside the
virtual environment, the `.pth` files in `.venv/lib/python3.13/site-packages/`
were given the macOS `hidden` flag (iCloud-synced folders do this), and
Python 3.13 skips hidden `.pth` files. Run
`chflags nohidden .venv/lib/python3.13/site-packages/*.pth`, prefix commands
with `PYTHONPATH=src`, or keep the repository outside iCloud.