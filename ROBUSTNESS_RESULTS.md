# Robustness results — Kai

Re-run on `dev-robustness` on 13 October 2026 after merging main at `5b3243f`.
Uses the merged notebook, its paired-market exhibits, and the committed data.
The executable source is `notebooks/report.ipynb`, Section 6 and Appendix A. This note
is a snapshot of that run, not a separate model implementation.

## Scope and sample


We keep the returns, risk-free rates, factors and estimation dates used above,
and replace ordinary OLS standard errors with Newey–West standard errors.
Coefficients and fitted returns do not change. The table counts individual
alphas with two-sided **p < 0.05**, using ordinary inference and HAC at **0, 3
and 6 monthly lags**. Three lags is the baseline; zero is the HC1
heteroskedasticity-only check. Bartlett weights and the T/(T-p) covariance
correction are used, with approximate Student-t inference on T-p residual
degrees of freedom, where p includes the intercept.

Stage 2 retains its local value-weighted proxy and 119-month window. Stage 3
retains its own common window: **106 months for US** and **107 for Singapore**.
The US Stage-3 models all use the Fama–French market factor. Comparisons across
HAC lags hold every input fixed; Stage-2 and Stage-3 CAPM rows are different
specifications. Assertions below verify the original coefficients, OLS p-values
and sample dates before interpreting the robust tests.

## Significant-alpha counts

Numbers below count individual two-sided p-values below 0.05. Alphas and ordinary
OLS results agree with the existing report; only the inference changes.

| Stage | Market | Model | Months | OLS | HAC(0) | HAC(3) | HAC(6) |
|---|---|---|---:|---:|---:|---:|---:|
| Stage 2 | US | CAPM (local index) | 119 | 2 | 2 | 2 | 2 |
| Stage 3 | US | CAPM | 106 | 1 | 1 | 1 | 3 |
| Stage 3 | US | FF3 | 106 | 1 | 2 | 2 | 4 |
| Stage 3 | US | FF3 + momentum | 106 | 1 | 1 | 2 | 2 |
| Stage 3 | US | Self-built SMB + WML | 106 | 1 | 1 | 1 | 2 |
| Stage 2 | SG | CAPM (local index) | 119 | 6 | 7 | 7 | 7 |
| Stage 3 | SG | CAPM | 107 | 8 | 9 | 8 | 8 |
| Stage 3 | SG | Self-built SMB + WML | 107 | 4 | 4 | 4 | 5 |

## Interpretation

**Interpretation.** At three lags, the broad pattern survives, but the identity
and strength of individual anomalies need qualification:

- **Stage 2:** the US count stays at 2/30 (DIS and XLY), while Singapore rises
  from 6/29 to 7/29 because Y92 becomes significant (ordinary p = 0.0714;
  HAC(3) p = 0.0199). Robust standard errors can decrease as well as increase.
- **US Stage 3:** NVDA remains significant in all four models at three lags.
  Its FF3 alpha is **2.40% per month**, with HAC(3) p = **0.0281**; it remains
  significant at six lags (p = 0.0442). However, its CAPM and FF3-plus-momentum
  p-values rise to **0.0600** and **0.0585** at six lags. Its positive estimated
  alpha is persistent, but significance is not uniform across specifications.
  XLK also becomes significant under FF3 and FF3 plus momentum at three lags
  (p = 0.0340 and 0.0472); the latter is close to the threshold.
- **Singapore Stage 3:** size and momentum reduce the significant count from
  **8 to 4 at three lags**, supporting the existing conclusion that factors
  explain part of the anomaly. At six lags the count falls from 8 to 5. Equal
  counts can hide different names: under CAPM, HAC(3) drops C07 and adds Y92
  relative to ordinary inference. The factor-model count reaches five at six
  lags because G13 becomes significant.

These results qualify the report's ordinary-inference conclusions; they do not
establish profitable trading opportunities or validate the existing joint tests.
Per-asset p-values for every lag, including all borderline cases, are saved in
`outputs/inference_robustness/inference_flags.csv`.

## Stocks-only check


This additional check removes the nine sector ETFs from the response assets,
leaving 21 stocks, while keeping Stage 3's existing factors, risk-free rate and
106-month calendar. It changes the set of assets being summarised, **not** the
regression of any retained stock. It does not rebuild the market, self-built
factors, portfolio weights or joint test.
**Interpretation.** At three lags, NVDA is the only significant stock in each
US Stage-3 model. The extra FF3 and FF3-plus-momentum flag in the 30-asset
analysis is XLK, which is excluded by construction here. Mean absolute monthly
FF3 alpha is **0.50% for the 21 stocks versus 0.40% for all 30 assets**. The
lower full-universe average partly reflects asset composition; removing ETFs
does not improve the estimated alpha of any retained stock.

## Limitations


- **Individual versus joint inference.** These HAC tests do not replace or
  validate the report's classical GRS tests. A failure to reject is not proof
  of zero alpha. No HAC-robust joint-alpha test has been run.
- **Small samples and lag choice.** Only 106–119 monthly observations are
  available, and the Student-t reference is an approximation for HAC. The
  prechosen 0/3/6 checks expose sensitivity; they must not be used to select
  whichever lag produces the preferred significance result. HAC does not
  correct omitted factors or changing coefficients.
- **Multiple testing.** The 5% screens are unadjusted across assets and models.
  Correlated stocks and overlapping sector ETFs make individual flags poor
  evidence of independent discoveries. Report effect sizes and uncertainty,
  especially for p-values near 0.05.
- **Data and sample conventions.** These runs inherit survivorship bias,
  incomplete historical share counts and the existing self-built factors.
  The US Stage-2 August 2026 RF is the pipeline's **0.003 monthly fallback**,
  because the FF file stops in July; US Stage 3 excludes that month. The cached
  INTC April 2026 return of approximately **+114%** remains unverified. HAC
  changes sampling uncertainty; it does not repair data errors.
- **Benchmark timing.** The existing local-index code multiplies same-month
  cap weights by that month's returns. If those weights contain month-end
  prices, they embed information unavailable at the start of the month. These
  runs preserve that specification for comparability; a separate lagged-weight
  check is needed before relying on its investable benchmark performance.

**Report conclusion.** The Singapore factor-absorption result survives the
baseline robust check, while several US individual-alpha claims are sensitive
to lag and model choice. Treat these as qualifications to the inference, not a
new portfolio recommendation or a robust validation of the GRS conclusions.

## Reproduce and inspect

Run `notebooks/report.ipynb` from top to bottom using the existing project
environment (`uv sync --extra notebook`, then `uv run jupyter lab`).
The notebook exports detailed estimates, p-values, exact sample dates, subset
summaries and a manifest of data hashes and package versions under
`outputs/inference_robustness/`. Generated files are git-ignored. This check
uses cached data and does not download or revise any input series.
