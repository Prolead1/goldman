# Kai's individual portfolio

This branch adds a self-contained Stage 1 study using the **21 individual US stocks**
already in the shared data. All nine sector ETFs are excluded by instrument type.
The shared downloaders, raw data and original notebook are unchanged.

## Start here

With the project dependencies installed, run from the checkout:

```sh
python run_kai_us21.py
```

If using the project's documented uv workflow:

```sh
uv run python run_kai_us21.py
```

Alternatively, run `notebooks/kai_us21.ipynb` from top to bottom.
These entry points run offline and write only to `outputs/kai_us21/`.

Open **`outputs/kai_us21/meeting_report.md`** for the findings and charts.
Bring the report, `weights.csv`, and `comparison_holdout.csv` to the meeting.
Share the entire output folder to keep image links working. The folder is git-ignored,
so others need to regenerate it or receive it separately.

## Fixed choices

- Assets: AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, JNJ, PG, XOM,
  CVX, KO, PEP, WMT, HD, DIS, CSCO, INTC, MRK, VZ.
- Sample: 118 monthly returns, October 2016 through July 2026.
- RF: observed supplied FF3 values, with no fallback month.
- Methods: unrestricted tangency; value weighting over these same 21 assets;
  monthly rebalanced equal weighting. FF US market is an additional reference.
- Historical value-weighted returns use the previous calendar month's weights.
- Stability: two consecutive 59-month estimation samples.
- Extra holdout: fit tangency once on the first 96 months, then evaluate the next 22.
  This is a diagnostic, not a claim of unbiased investable performance.

## Reading the results

Full-sample tangency results are fitted with hindsight. The holdout uses a different
weight vector, fitted on training data only. Short positions and gross exposure can
be large; returns exclude transaction, borrowing and financing costs.

The inherited historical market-cap estimates use some future share counts for missing
early history. The supplied stock selection has survivorship bias. An unusually large
Intel monthly return is flagged in the report; confirm it before drawing strong conclusions.

The return/weight CSVs retain precision. Report tables are rounded for readability.
`manifest.json` records input hashes and the runtime package versions used for each run.

## Validation

The full suite passed with 45 tests and 93.50% coverage when this analysis was added;
lint checks passed. New checks cover prior-month weights, long/short return arithmetic,
drawdown from initial capital, training-only holdout weights, and complete observed RF.
The report and all three generated charts were inspected.

To rerun checks in the project environment:

```sh
uv run pytest
uv run ruff check .
```

AI assistance with code, validation and drafting should be acknowledged in the final
appendix. Group members remain responsible for verifying the inputs and interpreting results.

## Stage 2: CAPM

Run `python run_kai_stage2.py` (or `uv run python run_kai_stage2.py`),
or open `notebooks/kai_stage2.ipynb` and run its cells top to bottom.
The existing cached data and Stage 1 selection/window are reused. No new dependencies
or data downloads are required beyond the project's existing analysis environment.

Read `outputs/kai_us21_stage2/stage2_report.md`. It includes all four required exhibits:

- **2a:** each stock's beta, alpha, classical alpha t-statistic, and R-squared.
- **2b:** empirical SML scatter with fitted and theoretical lines.
- **2c:** the handout's permitted chance-flag assessment of the zero-alpha hypothesis.
  This is explicitly not a formal GRS test or a global p-value.
- **2d:** fitted slope and intercept versus CAPM, with interpretation.

Extra checks use Newey-West standard errors and leave-one-stock-out SML fits.
The full-precision regression CSV, aligned input returns, residuals, line coefficients,
alpha assessment, charts and input/version manifest are saved alongside the report.
Stage 1 outputs remain separate. Generated outputs are git-ignored.

The main analysis retains all 21 assets, including the unverified Intel observation
flagged in Stage 1. Stage 3 factor regressions are described below; Stage 4 market-proxy
robustness and the final memo remain outstanding. Acknowledge AI assistance with this addition too.

## Stage 3: size and value factors

Run `python run_kai_stage3.py` (or `uv run python run_kai_stage3.py`),
or run `notebooks/kai_stage3.ipynb` top to bottom. The same 21 assets and 118 months are
used for matched CAPM and Fama-French three-factor regressions. No new downloads or
analysis dependencies are required.

Open `outputs/kai_us21_stage3/stage3_report.md` for:

- **3a:** sourced construction descriptions of the published SMB and HML factors.
- **3b:** before/after alpha table and chart, including classical and HAC uncertainty.
- **3c:** which Stage 2 flags persist, attenuate or newly appear, with explicit criteria.

The folder also contains all factor loadings and t-statistics, residuals, adjusted R-squared
comparisons, a mean-return decomposition, HAC lag sensitivity, and an input/version manifest.
It is git-ignored; regenerate it or share the folder separately with its images.

The full suite passed with 56 tests and 93.21% coverage when Stage 3 was added; lint checks
passed and the generated charts were inspected. Stage 4 remains. The Intel data-quality
flag, sample-selection limitations and AI acknowledgment requirement still apply.
