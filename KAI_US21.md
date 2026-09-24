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
