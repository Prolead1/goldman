# goldman - FE5108 project

**From the Mutual-Fund Theorem to the Security Market Line: does the theory survive real data?**

A reproducible research pipeline for the project, organized by analysis:
mean–variance portfolio choice and estimation error, the CAPM and the empirical
security market line, factor models, and the investment-committee memo. The
portfolio-choice (Stage 1), CAPM (Stage 2) and factor (Stage 3) analyses are
implemented and tested for **both** the US and Singapore universes; the Stage-4
investment-committee memo is written in `notebooks/report.ipynb`.

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
│   │   ├── cleaning.py         # SG total returns + caps rebuilt from SGX data
│   │   └── fetch.py            # explicit, network-only fetchers (not imported)
│   ├── analysis/               # portfolio construction and exhibits
│   │   ├── portfolio.py        # tangency weights, 1/N, subsample re-estimation
│   │   ├── portfolio_choice.py # summary stats, weight chart, subsample drift
│   │   ├── capm.py             # SCL regressions, empirical SML, GRS test
│   │   ├── factor_construction.py  # self-built SMB/WML/LOWVOL/REV + tradability
│   │   ├── industry_momentum.py  # momentum on 10 Ken French industries, 1964-2026
│   │   ├── factor_models.py    # multi-factor regressions, alpha comparison
│   │   ├── robustness.py       # no-short and shrinkage variants
│   │   └── out_of_sample.py    # expanding-window walk-forward validation
│   ├── viz.py                  # shared theme (seaborn + qbstyles) and figures
│   └── __init__.py             # public API
├── notebooks/
│   └── report.ipynb            # the report + reproducible entry point
├── tests/                      # pytest suite for the math and loaders
├── scripts/
│   └── audit_yahoo_prices.py   # network check of the price files vs raw Yahoo data
├── data/                       # raw cached CSVs (committed)
│   ├── raw/<universe>/        # per-ticker prices + shares cache
│   ├── raw/sg_daily/          # SG daily cache: Yahoo close/dividends + SGX price
│   └── raw/fx/                # month-end USD/SGD
└── outputs/                    # generated exhibits (git-ignored)
```

## Quickstart

```bash
uv sync --extra notebook    # env + Jupyter
uv run jupyter lab notebooks/report.ipynb   # run all cells
uv run pytest               # tests + coverage gate (min 80%)
uv run ruff check .
```

`notebooks/report.ipynb` is **both the written report and the reproducible entry
point**: it is organised as a report (executive summary, methodology, Stage 1,
Stage 2, Stage 3, Roll robustness, and the Stage-4 memo, which carries the
out-of-sample validation and the Roll exhibit), closing with an appendix for the
contribution statement, AI-tool acknowledgement, data sources, references and
reproducibility. Every table and figure in it
is generated from the committed data when the notebook is run top to bottom. The primary universe is the US (`PRIMARY = "us"`); Singapore is carried
throughout as a cross-market check. Code cells are tagged `hide_input`, so a
report export can hide them, e.g.
`uv run jupyter nbconvert --to html --no-input notebooks/report.ipynb`.

Network fetchers need the optional `data` extra and open internet; they run
only when invoked — nothing imports them:

```bash
uv sync --extra data
uv run python -m goldman.data.fetch prices       # prices + FF3 factors
uv run python -m goldman.data.fetch caps         # fetch + build value weights
uv run python -m goldman.data.fetch sg-daily     # SG daily cache (Yahoo + SGX) + USD/SGD
uv run python -m goldman.data.fetch sg-build     # offline: clean SG prices + weights
uv run python -m goldman.data.fetch long-history # Ken French 10 industries + factors since 1926
uv run python -m goldman.data.fetch all          # everything
```

The Singapore files are not Yahoo's adjusted closes (see *Singapore data
cleaning* below). `sg-daily` caches Yahoo's daily close and dividends next to
SGX's own unadjusted last-traded price, and `sg-build` (offline, repeatable)
rebuilds `data/prices_sg_monthly.csv`, `data/value_weights_sg.csv` and
`data/cleaning_report_sg.txt` from that cache and `data/corporate_actions_sg.csv`.
A test checks that the committed files reproduce from the cache exactly.
`uv run python scripts/audit_yahoo_prices.py` re-runs the audit against live
Yahoo data.

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
| `analysis/factor_construction.py` | Four self-built long–short factors: SMB (lagged point-in-time size), WML (12-1 momentum), LOWVOL (trailing 12-month volatility) and REV (1-month reversal); turnover, cost sensitivity and drawdown of all four (`self_factor_tradability`) | `factor_cumulative_<universe>.png`, `factor_tradability.csv` (US panel plots the downloaded Fama–French SMB/HML) |
| `analysis/industry_momentum.py` | Industry momentum (Moskowitz and Grinblatt 1999) on the 10 Ken French industries, 1964-07 to 2026-08: premium, CAPM and FF3 alphas, correlation with Ken French `Mom`, turnover and break-even cost, by sub-period | `industry_momentum.png` |
| `analysis/factor_models.py` | Multi-factor SCL regressions; alphas before/after factors on a common sample; joint GRS | `factors_<universe>_alphas.csv`, `..._summary.csv`, `factors_<universe>_<model>_scl.csv`, `factors_<universe>_alpha_comparison.png` |
| `analysis/robustness.py` | No-short maximum Sharpe, shrinkage of covariance and means, half sample stability, out of sample Sharpe | `robustness_<universe>_diagnostics.csv`, `..._weights.csv`, `robustness_<universe>.png` |
| `analysis/out_of_sample.py` | Expanding-window walk-forward backtest: re-estimate the index, 1/N and the tangency variants each month, trade the next month; out-of-sample growth and Sharpe | `out_of_sample_<universe>_growth.png`, `out_of_sample_<universe>_sharpe.png` |
| `data/markets.py` | Market proxy `R_M - R_f`: cap-weighted composite or FF `Mkt-RF` | — |
| `viz.py` | The report's look (seaborn with the QuantumBlack dark theme) and one figure per story beat: correlation, risk/return, weights, drift, robustness, alphas, SML, GRS, proxy comparison, factor growth, factor tradability, alpha comparison | `*.png` exhibits |

Adding an analysis means adding a module under `src/goldman/analysis/` and a
section in `notebooks/report.ipynb`; the notebook reproduces every stage top to
bottom.

## Data

| File | What it is |
|---|---|
| `data/prices_us_monthly.csv` | 30 US names (21 mega-caps + 9 SPDR sector ETFs) |
| `data/prices_sg_monthly.csv` | ~30 large SGX names (approximately the STI), SGD total-return index rebuilt by `goldman.data.cleaning` |
| `data/corporate_actions_sg.csv` | Every SG correction (spin-offs, distributions in specie, share-count overrides) with its SGX source |
| `data/cleaning_report_sg.txt` | What each SG correction changed, per ticker, versus Yahoo's adjusted close |
| `data/raw/sg_daily/<ticker>.csv` | Daily Yahoo `close`, `adj_close`, `dividend`, `split` and SGX `sgx_close`, `currency` |
| `data/raw/fx/usdsgd_monthly.csv` | Month-end USD/SGD (Yahoo `SGD=X`), for the three USD-quoted SG listings |
| `data/ff3_monthly.csv` | Fama–French 3 factors + RF, decimals, monthly (US); the Stage-3 external benchmark |
| `data/long_ind10_vw_monthly.csv` | Ken French 10 value-weighted US industry portfolios, 1963-07 onward, decimals (every CRSP stock, no survivorship) |
| `data/long_ff3_monthly.csv`, `data/long_momentum_monthly.csv` | Ken French FF3 + RF (1926-07 onward) and `Mom` (1927-01 onward), decimals; the benchmarks for industry momentum |
| `data/rf_sg_monthly.csv` | Monthly SG risk-free from the MAS compounded SORA Index (month-over-month change), decimal. Raw export and method in `data/raw/mas/` and `rf_sg_monthly.source.txt` |
| `data/raw/<universe>/<ticker>.csv` | Raw per-ticker `close`, `adj_close`, `shares` cache |
| `data/value_weights_us.csv` | US point-in-time market cap -> monthly value weights |
| `data/value_weights_sg.csv` | SG point-in-time market cap -> monthly value weights |
| `data/sample_synthetic_prices.csv` | **Fake** data for smoke-testing; not for the report |
| `data/coverage_report.txt` | What the downloader fetched |

Cleaning decisions (sparse-ticker drops, fully-observed months, the Singapore
rebuild) are made once, in `src/goldman/data/`.

### Honest caveats

- US prices are Yahoo **adjusted closes** (splits + dividends), a total-return
  proxy. Rebuilt from Yahoo's raw closes plus dividends they agree within 0.11%
  in every month, with no mis-adjusted ex-date.
- **Singapore data cleaning.** Yahoo's adjusted close is wrong for several SGX
  names, so `goldman.data.cleaning` rebuilds every SG series:
  - ThaiBev (Y92): before 2024 Yahoo adjusts each dividend using the baht amount
    as if it were SGD (about 26x too large); its adjusted price grows 287x over
    the decade against a true total return of 0.67x.
  - Jardine C&C (C07): since 2020-09 Yahoo adjusts with the US-dollar amount,
    so each dividend counts about 78%.
  - Distributions in specie (U96 2020-09, BS6 2022-04, BN4 2023-02, plus the
    small C09 2022-05 and BN4 2023-10/2026-04 ones): Yahoo scales earlier
    closes but not earlier dividends, inflating pre-event yields up to 2x. For
    the three large spin-offs it also shows an ex-date jump of +13% to +22%
    where holders gained +3% to +10%, so those ex-dates are rebuilt as
    (SGX ex price + dividend + distributed shares at their SGX price) / SGX cum price.
  - Market caps use SGX's unadjusted month-end price, since Yahoo's scaled
    close halves Sembcorp's and Yangzijiang's pre-spin-off caps. Two share-count
    glitches are overridden (UOB 2017-03; Jardine Matheson 2019-07..2022-05,
    where Yahoo counts the Jardine Strategic cross-holding).
  - DFI Retail, Hongkong Land and Jardine Matheson trade in USD on SGX. Their
    returns and caps are converted to SGD at month-end USD/SGD.
  - Every correction has its SGX corporate-action number in
    `data/corporate_actions_sg.csv`. 9CI (listed 2021) is cleaned but still
    dropped by the 90-month rule.
  - SGX's historical-price service returns roughly the last ten years, and the
    committed cache starts in 2016-09. A later `sg-daily --refresh` would lose
    the first months of the panel, so keep the committed cache for this report.
- One known US approximation: Yahoo records Merck's 2021 Organon spin-off as a
  1.048 split, so MRK's market cap before June 2021 is about 5% too high in the
  value weights. Its returns are unaffected.
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
  `ff_market` proxy is unavailable for SG. Stage 3 therefore **builds four
  long–short factors from the universe itself** for both markets — size,
  momentum, low volatility and short-term reversal — and uses Ken French only as
  the US external benchmark. The self-built factors are equal-weighted 30/40/30
  quantile spreads; a mega-cap universe makes the size leg a weak proxy, which
  the report states.
- Residual data limitations: Yahoo has no early share history for a handful of
  names (META before 2022-07; a few SG listings), so those months carry the
  earliest known count back to the sample start — an approximation, but a
  point-in-time price times a plausible share count rather than today's cap.
  The weight chart uses value weights by default; set `comparison=None` in the
  notebook to use the 1/N comparison instead.

## Reproducibility

`notebooks/report.ipynb` regenerates every exhibit from the committed raw CSVs
when run top to bottom, and doubles as the report. Generated exhibits go to
`outputs/` (git-ignored; US and SG portfolio exhibits under `outputs/us/` and
`outputs/sg/`, CAPM exhibits as `capm_<universe>_<proxy>_*`). Package versions
are pinned in `pyproject.toml` / `uv.lock`.

### Alpha-inference sensitivity in the memo

Section 7 also runs `analysis/robust_inference.py` and `analysis/sensitivity.py`
against the existing Stage-2/3 model inputs. Exhibit 5 compares ordinary and
Newey-West standard errors at 0, 3 and 6 monthly lags; the takeaway reports
those counts. Per-asset p-values, summary counts, exact
sample dates, data hashes and package versions are saved under
`outputs/inference_robustness/`. To verify a summary count, count the per-asset
p-values below 0.05 in `inference_flags.csv` for the corresponding model and
setting. Running the notebook top to bottom reproduces this addition without
downloading data or adding report appendices.
