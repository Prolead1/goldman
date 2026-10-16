# Robustness results — Kai

Re-run on 16 October 2026 on `dev-robustness`, after merging main at `62624d5`.
This is a supporting repository note, not an appendix to the submitted report.
The executable analysis and concise interpretation are in Section 7 of
`notebooks/report.ipynb`, under **How much confidence should we place in individual alphas?**

## Results

Counts of individual, two-sided alpha p-values below 0.05. The US contains 30
assets; Singapore contains 29. Compare across rows only when model and sample
differences are intended: Stage 2 uses local indices and 119 months; Stage 3 uses
the Fama-French market factor for the US (106 months) and the local index for
Singapore (107 months). Exact start/end dates are in the exported summary.

| Stage | Market | Model | Months | OLS | HAC(0) | HAC(3) | HAC(6) |
|---|---|---|---:|---:|---:|---:|---:|
| Stage 2 | US | CAPM (local index) | 119 | 2 | 2 | 2 | 2 |
| Stage 3 | US | CAPM | 106 | 1 | 1 | 1 | 3 |
| Stage 3 | US | FF3 | 106 | 1 | 2 | 2 | 4 |
| Stage 3 | US | FF3 + momentum | 106 | 1 | 1 | 2 | 2 |
| Stage 3 | US | Self-built SMB + WML | 106 | 1 | 1 | 1 | 2 |
| Stage 2 | SG | CAPM (local index) | 119 | 3 | 4 | 4 | 4 |
| Stage 3 | SG | CAPM | 107 | 5 | 4 | 3 | 5 |
| Stage 3 | SG | Self-built SMB + WML | 107 | 5 | 5 | 3 | 6 |

**Takeaway.** Singapore's Stage-3 counts before and after adding size and momentum are 5 to 5 under OLS; 3 to 3 under HAC(3); 5 to 6 under HAC(6). Adding factors does not consistently reduce the significant-alpha count. In the US, NVDA is significant in 4 of 4 Stage-3 models at three lags and 2 at six. Equal counts can hide different assets. For the committee, this qualifies confidence in individual alpha signals; it does not provide a new allocation recommendation.

## Method and limits

Returns, factors, risk-free rates and model dates are inherited from the existing
report. HAC changes standard errors, not estimated alphas or portfolio returns.
We use Bartlett weights, the T/(T-p) covariance correction and approximate
Student-t(T-p) p-values, with p including the intercept. HAC(0) is the HC1
heteroskedasticity-only case; three lags is the baseline and zero/six expose lag
sensitivity. These settings are not optimised to obtain significance.

Individual screens are unadjusted for multiple testing and do not replace the
classical joint GRS tests. Equal counts need not refer to the same assets. The
sample is short for inference, and HAC cannot fix data errors, benchmark timing,
survivorship bias or omitted factors. No stocks-only comparison or extra
technical appendix is included in the memo.

Method source: Newey, W. K., and West, K. D. (1987), *A Simple, Positive
Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
Matrix*, Econometrica 55(3), 703–708. https://doi.org/10.2307/1913610
Data sources and cleaning conventions are those documented in the main report
and README; this addition does not download or alter input data.

## Reproduce and verify

Use the project's normal environment (`uv sync --extra notebook`), open
`notebooks/report.ipynb` with `uv run jupyter lab notebooks/report.ipynb`, then
restart the kernel and run all cells. The robustness cells reuse the preceding
Stage-2 and Stage-3 objects, and assert agreement with their coefficients,
ordinary p-values and exact sample dates.

The notebook generates these files under `outputs/inference_robustness/`:

- `inference_summary.csv`: the memo table, with sample dates and asset counts.
- `inference_flags.csv`: per-asset alphas and ordinary/HAC p-values; counting
  p-values below 0.05 reproduces every summary count.
- `inference_details.csv`: full regression and uncertainty estimates.
- `stage_*_sample.csv`: exact estimation calendars.
- `memo_takeaway.md`: the numerical interpretation generated from this run.
- `manifest.json`: data SHA-256 hashes, package versions and inference settings.

Verification: all 33 code cells completed in order with no recorded
errors; all original main cell sources were preserved; all eight table rows
were independently recounted from the per-asset CSV. The 41 existing robustness
and sensitivity tests passed. No full-suite rerun was needed for this notebook
integration; the unchanged modules were covered by the earlier 176-test run.
The execution environment's numerical package versions are recorded in the
manifest; this verification used a temporary environment rather than `uv.lock`.
Final PDF pagination remains to be checked when the group assembles its report.
