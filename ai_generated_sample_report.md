# FE5108 Reference Report — From the Mutual-Fund Theorem to the Security Market Line

Oct 9, 2026 · @Aryan

## About this reference

This is an AI-generated example of the final report, built from the outputs of `notebooks/portfolio_analysis.ipynb` on branch `dev-aryansethi-stage-2-3`. Use it as a model for structure, exhibits and argument; the submitted PDF must be written in the group's own words.

- Every number comes from the committed data (US universe, 2016-10 to 2026-08). Re-run `uv run python run_all.py` and re-check each figure before submitting.
- Do not re-download data before submitting: Ken French and Yahoo revise history, which would change the numbers.
- The handout requires a one-sentence AI acknowledgement in the appendix and a signed contribution statement.
- Length guide: the PDF may have 12 pages plus an appendix. Sections 1–6 below map to those 12 pages; section 7 is the appendix.

## 1. Executive summary

**Recommendation: hold the value-weighted market index, and set risk with the split between the index and T-bills.** The textbook optimal portfolio cannot be estimated well enough to beat it.

**Question.** Does the chain from mean–variance choice and the Mutual-Fund Theorem to the CAPM and factor models survive ten years of data? We test it on 30 large US assets (21 mega-cap stocks, 9 sector ETFs), monthly, October 2016 to August 2026. Market proxy: the CRSP value-weighted index. Risk-free rate: the one-month T-bill.

**Finding 1 — the estimated optimal portfolio is mostly noise.** The tangency portfolio holds up to 272% of capital in one asset and shorts 13 others. Estimated on each half of the sample, its weights correlate only 0.35. A no-short constraint or shrinkage toward the CAPM cuts positions to 17–46%, but the two halves still disagree.

**Finding 2 — the CAPM is not rejected, mainly because ten years cannot reject it.** One alpha (NVDA, +35% a year) has |t| > 2, against 1.4 expected by chance. The GRS joint test gives p = 0.21 (0.07 on the stocks alone). An alpha must exceed about 16% a year to be detectable. Our SML looks steeper than theory only because of NVDA; on survivorship-free portfolios it is flatter than theory with a positive intercept.

**Finding 3 — factors explained risk, not return, this decade.** Size, value and momentum earned roughly zero in 2016–2026, so they raise R² from 0.37 to 0.48 without moving alphas. On sorted portfolios they work as designed. Our self-built industry-momentum factor is priced since 1964 (FF3 alpha 4.9%, t = 2.8) but costs about as much to trade as it earns.

**Finding 4 — every CAPM verdict depends on the proxy.** Against MSCI World, four alphas are significant instead of one and the GRS p-value falls to 0.10. Against the in-sample tangency portfolio, every alpha is exactly zero.

**The evidence for the recommendation.** In an out-of-sample race on our assets since 2021, and on ten US industries since 1968, no rule beats the index significantly. The plug-in tangency portfolio is significantly worse and loses everything. Constrained or shrunk versions are safe but trail the index after costs. A moderately risk-averse investor (γ = 3) should hold about 80% in the index and the rest in T-bills.

**Result we trust least:** the ranking of the sensible strategies on our own universe — 59 months of a tech-led bull market on a list chosen with hindsight.

## 2. Data and setup

The sample is 119 months of returns, October 2016 to August 2026, on 30 US assets: 21 mega-cap stocks and the 9 SPDR sector ETFs with ten years of history. All returns are in US dollars.

| Input | Choice | Why |
| --- | --- | --- |
| Asset returns | Yahoo Finance monthly adjusted closes (splits and dividends) | A total-return proxy, free and reproducible |
| Market proxy | Ken French `Mkt-RF`: value-weighted CRSP stocks on NYSE, AMEX and NASDAQ | The broadest US equity portfolio, value-weighted like the CAPM market, and from the same source as the Stage 3 factors |
| Risk-free rate | Ken French one-month T-bill | The standard short-term US rate; the same series as the factors |
| Value weights | Split-unadjusted close × shares outstanding, shares carried forward from report dates | Point-in-time: no month uses a future share count |
| Factors | Ken French SMB, HML, Mom; RMW and CMA for robustness | The standard US factors, built from every CRSP stock |
| Test portfolios | Ken French 25 size × book-to-market and 10 momentum portfolios | Built from all stocks, so free of our selection |
| Stage 4 extras | S&P 500 (SPY) and MSCI World (URTH) ETFs; Ken French developed markets; 10 industries since 1963; market since 1926 | Alternative proxies (Roll) and a long, survivorship-free test |

**Cleaning decisions.**

- Monthly bars carry the month-end close under a month-start label, so the return labelled October is October's return. It is matched to Ken French's October values by calendar month.
- Tickers with fewer than 90 monthly observations are dropped (none in the US universe). Only months where every asset has a return are kept.
- Regressions use only months covered by the factor file; no risk-free rate is filled in.
- A value-weighted investor holds the weights known at the end of the previous month.
- Sector ETFs enter the value weights with their fund size (1.4% of the index in total). They also hold the stocks, so those stocks are slightly double-counted.
- All data are cached as CSV files (Ken French release 2026-08). The analysis runs offline from one notebook.

**Survivorship bias.** The universe is today's largest companies, backfilled ten years. Firms that shrank, merged or failed are missing, so average returns are biased upward. Each stage below says how this bias cuts, and Stages 2–4 repeat the key tests on survivorship-free Ken French portfolios.

## 3. Stage 1 — Portfolio choice

The estimated tangency portfolio is dominated by estimation error. It takes extreme long–short positions that change completely between subsamples. A no-short constraint or shrinkage tames the size of the positions, not their instability.

### (1a) Summary statistics

**Table 1 — Annualised statistics, Oct 2016 – Aug 2026.** *Ten years of mega-cap returns: average returns of 7–61% a year, volatility of 13–53%, Sharpe ratios from 0.15 (DIS) to 1.25 (NVDA).*

| Asset | Mean (% p.a.) | Vol (% p.a.) | Sharpe |
| --- | --- | --- | --- |
| NVDA | 60.7 | 46.7 | 1.25 |
| XLK | 24.1 | 21.0 | 1.04 |
| MSFT | 25.9 | 22.9 | 1.03 |
| AAPL | 29.2 | 27.2 | 0.98 |
| GOOGL | 25.3 | 27.0 | 0.85 |
| WMT | 18.5 | 19.2 | 0.84 |
| JPM | 22.6 | 24.3 | 0.84 |
| AMZN | 23.3 | 31.2 | 0.67 |
| CSCO | 18.6 | 24.9 | 0.66 |
| XLI | 14.6 | 19.1 | 0.65 |
| XLF | 14.8 | 19.4 | 0.64 |
| BAC | 20.2 | 28.6 | 0.63 |
| XLV | 11.4 | 14.6 | 0.62 |
| KO | 11.9 | 16.4 | 0.59 |
| JNJ | 12.4 | 17.3 | 0.58 |
| MRK | 14.5 | 21.4 | 0.57 |
| XLY | 13.9 | 20.4 | 0.57 |
| META | 21.7 | 35.7 | 0.54 |
| HD | 14.5 | 22.9 | 0.53 |
| XLB | 11.7 | 18.6 | 0.50 |
| XLU | 9.8 | 15.3 | 0.49 |
| CVX | 15.3 | 28.5 | 0.45 |
| XLP | 8.2 | 13.1 | 0.45 |
| XOM | 14.6 | 28.5 | 0.43 |
| XLE | 14.6 | 30.2 | 0.41 |
| PG | 9.0 | 17.0 | 0.40 |
| INTC | 22.4 | 53.2 | 0.38 |
| PEP | 6.9 | 16.1 | 0.28 |
| VZ | 6.8 | 19.0 | 0.24 |
| DIS | 6.8 | 30.6 | 0.15 |

Means and volatilities are of total returns (monthly × 12 and × √12). The Sharpe ratio is the mean excess return over the volatility of excess returns. Survivorship pushes these means up: the list is today's winners.

### (1b) Tangency weights against value weights

The tangency portfolio is w ∝ Σ̂⁻¹μ̂ₑ, scaled to sum to one, with μ̂ₑ and Σ̂ the sample mean and covariance of excess returns. The value-weighted portfolio uses point-in-time market caps; the chart shows their sample average.

&#91;image: Figure 1: tangency weights against value weights\]

**Figure 1.** *The estimated tangency portfolio bears no resemblance to the market: positions up to 272% of capital (short XLP), long and short, against value weights that never exceed 16% (AAPL 15%, MSFT 14%, GOOGL 11%).*

### (1c) Estimation error

We re-estimate the portfolio on each half of the sample. The two weight vectors correlate only 0.35, and single positions swing wildly.

**Table 2 — Tangency weights by subsample (selected assets).** *The same theory and assets give very different portfolios five years apart.*

| Asset | First half | Second half | Full sample |
| --- | --- | --- | --- |
| KO | 0.12 | 4.40 | 1.17 |
| BAC | −0.11 | −2.79 | −0.46 |
| XLP | −1.68 | −3.92 | −2.72 |
| AAPL | 0.12 | 1.24 | 0.43 |
| MSFT | 1.05 | −0.18 | 0.14 |
| PG | 1.07 | 0.16 | 0.93 |

The reason is the inverse covariance matrix. It turns small, noisy differences in estimated means into large offsetting bets. Ten years of returns pin down volatilities well but average returns badly: NVDA's mean of 61% a year has a standard error of about 15%.

**Taming the weights.** The handout asks how a no-short constraint or shrinkage changes this. We compare four versions, each estimated on the full sample and on both halves:

- **Plug-in:** sample μ̂ and Σ̂.
- **Long-only:** the same inputs, no short sales. A constraint acts as shrinkage of Σ̂ (Jagannathan and Ma 2003).
- **Shrinkage:** Ledoit–Wolf covariance, and means pulled toward the CAPM prediction π = β × market premium. The data set the pull (54% toward the prior). Full shrinkage would give exactly the value-weighted portfolio.
- **Shrinkage, long-only:** both.

&#91;image: Figure 2: four versions of the tangency portfolio against value weights\]

**Figure 2.** *A no-short constraint or shrinkage cuts the largest position from 272% to 17–46%, but the two half-sample portfolios still do not agree.*

**Table 3 — How much each fix tames the portfolio.**

| Portfolio | Max weight | Gross exposure | Shorts | Half-sample corr | Half-sample L1 distance |
| --- | --- | --- | --- | --- | --- |
| Plug-in tangency | 272% | 15.9× | 13 | 0.35 | 33.7 |
| Long-only | 20% | 1.0× | 0 | −0.04 | 1.8 |
| Shrinkage | 46% | 5.2× | 13 | −0.02 | 9.6 |
| Shrinkage, long-only | 17% | 1.0× | 0 | −0.03 | 1.7 |
| Value weights | 15% | 1.0× | 0 | — | — |

The L1 distance is the turnover needed to move from the first-half portfolio to the second-half one, in multiples of capital. The fixes cut it from 34× to under 2×. But the tamed weights are uncorrelated across halves: taming limits the damage estimation error can do; it cannot create information that is not in μ̂.

**Is it the ETFs?** The sector ETFs hold the stocks, so some assets are near-copies of others, which makes Σ̂ hard to invert. Dropping the ETFs cuts the condition number of the correlation matrix from 490 to 94 and the largest weight from 272% to 79%. Yet the 21-stock portfolio still runs 4.5× gross with six shorts, and its halves correlate only 0.14. The overlap amplifies the problem; it does not cause it (Appendix A1).

## 4. Stage 2 — The CAPM

The CAPM survives our tests, but mostly because ten years of data cannot reject it. Betas are precise; alphas are large but imprecise. Whether the security market line looks too steep or too flat depends on one stock and on survivorship.

For each asset we estimate the security characteristic line by OLS:

```latex
R_{i,t} - R_{f,t} = \alpha_i + \beta_i\,(R_{M,t} - R_{f,t}) + \varepsilon_{i,t}
```

### (2a) Betas, alphas and fit

**Table 4 — CAPM regressions, 119 months.** *Betas are precise (every t(β) > 2), alphas are not: only NVDA's alpha has |t| > 2 with OLS errors.*

| Asset | β | α (% p.a.) | t(α) | t(α) Newey–West | R² |
| --- | --- | --- | --- | --- | --- |
| NVDA | 1.75 | +35.2 | 2.88 | 2.47 | 0.36 |
| MSFT | 0.91 | +11.6 | 1.99 | 1.90 | 0.40 |
| XLK | 1.14 | +6.6 | 1.94 | 2.19 | 0.76 |
| AAPL | 1.10 | +12.2 | 1.79 | 2.04 | 0.42 |
| WMT | 0.49 | +9.6 | 1.68 | 1.86 | 0.17 |
| GOOGL | 1.06 | +8.9 | 1.29 | 1.44 | 0.40 |
| MRK | 0.26 | +8.7 | 1.26 | 1.19 | 0.04 |
| JPM | 1.07 | +6.1 | 1.09 | 1.16 | 0.50 |
| JNJ | 0.42 | +4.5 | 0.86 | 0.93 | 0.15 |
| KO | 0.41 | +4.2 | 0.84 | 0.82 | 0.16 |
| CSCO | 0.91 | +4.3 | 0.64 | 0.73 | 0.34 |
| AMZN | 1.27 | +4.1 | 0.53 | 0.53 | 0.42 |
| XLU | 0.42 | +1.8 | 0.41 | 0.50 | 0.19 |
| PG | 0.35 | +2.0 | 0.38 | 0.43 | 0.11 |
| META | 1.18 | +3.6 | 0.37 | 0.35 | 0.28 |
| XOM | 0.75 | +2.3 | 0.27 | 0.21 | 0.18 |
| XLV | 0.62 | +0.9 | 0.24 | 0.29 | 0.46 |
| CVX | 0.86 | +1.5 | 0.18 | 0.17 | 0.23 |
| INTC | 1.37 | +1.9 | 0.12 | 0.14 | 0.17 |
| VZ | 0.31 | +0.4 | 0.06 | 0.07 | 0.07 |
| BAC | 1.33 | +0.3 | 0.04 | 0.05 | 0.55 |
| HD | 0.95 | −0.5 | −0.09 | −0.11 | 0.44 |
| XLE | 1.03 | −1.4 | −0.17 | −0.13 | 0.30 |
| XLF | 0.99 | −0.7 | −0.18 | −0.17 | 0.67 |
| XLP | 0.50 | −0.7 | −0.20 | −0.23 | 0.37 |
| PEP | 0.43 | −1.1 | −0.23 | −0.26 | 0.18 |
| XLI | 1.05 | −1.5 | −0.51 | −0.59 | 0.77 |
| XLB | 0.95 | −3.2 | −0.90 | −1.01 | 0.66 |
| XLY | 1.15 | −3.7 | −1.27 | −1.52 | 0.81 |
| DIS | 1.30 | −12.8 | −1.74 | −2.05 | 0.46 |

Betas follow economic intuition: staples, pharma and telecom sit at 0.26–0.43, banks and tech near or above 1, NVDA highest. R² is higher for the diversified ETFs (up to 0.81) than for single stocks (as low as 0.04 for MRK). Newey–West errors (4 lags) allow for changing volatility and autocorrelation; they move t-statistics only slightly.

### (2b) The empirical security market line

&#91;image: Figure 3: empirical security market line, 30 assets\]

**Figure 3.** *Our SML looks steeper than theory only because of one point, NVDA (β = 1.75, 58% a year excess return).* Fitted: −0.7% + 18.1% × β. Theory: 0% + 13.3% × β, the line through the origin with slope equal to the average market excess return.

### (2c) Are all alphas zero?

Two answers point the same way.

- **Counting.** Each test has a 4.8% chance of |t| > 2 under the null, so 30 assets should give about 1.4 by chance. We see one; the chance of at least one is 0.77. NVDA's t = 2.88 is not significant after a Bonferroni correction for 30 tests (p = 0.14).
- **GRS test (Gibbons, Ross and Shanken 1989).** One F-test of α₁ = … = α₃₀ = 0 that allows for correlated residuals: GRS = 1.25, F(30, 88), p = 0.21. We cannot reject that all alphas are zero.

```latex
\mathrm{GRS} = \frac{T-N-K}{N}\,\frac{\hat\alpha'\hat\Sigma^{-1}\hat\alpha}{1 + \bar\mu'\hat\Omega^{-1}\bar\mu} \sim F(N,\,T-N-K)
```

Three caveats keep this from being a vote for the CAPM:

- **Power.** For the median asset, the smallest alpha a 5% test detects with 80% power is about 16% a year. NVDA's alpha of 35% has a standard error of 12%.
- **Correlated tests.** Residuals of overlapping assets are strongly correlated (XLE–XOM 0.93, BAC–XLF 0.85, XLK–AAPL 0.46), even though the average pair is near zero. With Newey–West errors four alphas cross |t| = 2 (NVDA, XLK, AAPL, DIS), but three of them are correlated tech exposures.
- **Without the ETFs** the 21 stock alphas give GRS = 1.58, p = 0.07: significant at 10%, not at 5%.

The honest verdict is weak evidence against the CAPM, not enough to reject it.

### (2d) The SML against theory

We fit the SML with Fama–MacBeth regressions: each month, returns on betas, then averaged. Their standard errors allow for correlation across assets; the Shanken correction adds the error in estimated betas. Theory says intercept 0 and slope equal to the market premium.

**Table 5 — Fitted SML against theory, by test assets and period.** *The textbook pattern — too flat, intercept too high — appears everywhere except in the full 30-asset set, and no deviation is significant.*

| Test assets | Assets | Months | Intercept (% p.a.) | t(intercept = 0) | Slope (% p.a.) | Market premium (% p.a.) | t(slope = premium) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 assets | 30 | 119 | −0.7 | −0.12 | 18.1 | 13.3 | 0.59 |
| Without NVDA | 29 | 119 | +4.5 | 0.78 | 10.7 | 13.3 | −0.32 |
| Without mega-cap tech | 23 | 119 | +5.9 | 1.05 | 6.9 | 13.3 | −0.79 |
| First half, 2016-10 to 2021-08 | 30 | 59 | +10.1 | 1.48 | 8.9 | 17.7 | −0.88 |
| Second half, 2021-09 to 2026-08 | 30 | 60 | +4.2 | 0.51 | 9.3 | 8.8 | 0.04 |
| FF 25 size × book-to-market | 25 | 120 | +8.7 | 1.05 | 3.1 | 13.2 | −1.00 |
| FF 10 momentum | 10 | 120 | +11.4 | 1.47 | 1.3 | 13.2 | −1.26 |

&#91;image: Figure 4: empirical SML on the Ken French 25 size and book-to-market portfolios\]

**Figure 4.** *On portfolios built from every US stock, the SML is almost flat: +8.7% + 3.1% × β against a theoretical 13.2% × β.*

**Interpretation.** High-beta assets have not earned proportionally more, as Black (1972) and Frazzini and Pedersen (2014) predict when investors face borrowing limits. Our universe hides this because it is today's mega-caps backfilled: high-beta names that failed are missing, and the one that won spectacularly pulls the line up. Still, no slope differs from the premium with |t| above 1.3, and no intercept differs from zero with |t| above 1.5. Ten years show the direction, not the size, of the failure.

## 5. Stage 3 — Factors

In 2016–2026 the extra factors explained risk, not return: size, value and momentum earned roughly nothing, so they raised R² without absorbing our alphas. On sorted portfolios, where the factors were designed to work, they absorbed the pricing errors they target.

### (3a) Factor construction

All factors come from Kenneth French's data library: value-weighted portfolios of all CRSP common stocks on NYSE, AMEX and NASDAQ, with NYSE breakpoints. *Check each description against the library's own pages before using it.*

- **SMB (small minus big).** Each June, stocks are split at the NYSE median market cap into small and big, and independently into three book-to-market groups at the 30th and 70th percentiles. SMB is the average of the three small portfolios minus the average of the three big ones.
- **HML (high minus low).** From the same six portfolios: the average of small-value and big-value minus the average of small-growth and big-growth. Book equity is from the fiscal year ending in the previous calendar year, divided by market equity at the previous December.
- **Mom (momentum).** Each month, stocks are split at the NYSE median size and at the 30th and 70th percentiles of their return from month t−12 to t−2. Mom is the average of the two winner portfolios minus the average of the two loser portfolios. Skipping month t−1 avoids short-term reversal.
- **RMW and CMA (robustness).** Robust minus weak operating profitability, and conservative minus aggressive investment, from the five-factor model. Its SMB is built from different sorts, so it is kept separate from the FF3 SMB.

A factor can only lower an alpha through its average return: adding it moves alpha by roughly (loading × factor premium). So the first question is what each factor paid.

**Table 6 — Factor premia, Oct 2016 – Aug 2026.** *Only the market paid; size, value, momentum, profitability and investment all earned less than |t| = 0.9.*

| Factor | Mean (% p.a.) | t(mean) | Vol (% p.a.) | Sharpe |
| --- | --- | --- | --- | --- |
| Mkt-RF | +13.3 | 2.61 | 16.0 | 0.83 |
| SMB | −2.3 | −0.75 | 9.6 | −0.24 |
| HML | 0.0 | 0.00 | 13.5 | 0.00 |
| Mom | +2.4 | 0.54 | 14.1 | 0.17 |
| RMW | +2.5 | 0.89 | 9.0 | 0.28 |
| CMA | −0.6 | −0.22 | 8.7 | −0.07 |

This is the decade of the well-documented "death of value" and a small-cap drought.

### (3b) Alphas before and after adding factors

We nest the models: CAPM, then Fama–French three-factor (+SMB, HML), then Carhart four-factor (+Mom); FF5 is a robustness check.

**Table 7 — Joint fit by model and test assets.** *On our 30 assets the factors raise R² but leave alphas unchanged; on sorted portfolios they cut alphas sharply.*

| Test assets | Model | GRS | GRS p-value | Mean \|α\| (% p.a.) | Max \|α\| (% p.a.) | \|t\| > 2 | Average R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 assets | CAPM | 1.25 | 0.21 | 5.2 | 35.2 | 1 | 0.37 |
| 30 assets | FF3 | 1.30 | 0.17 | 4.6 | 34.2 | 1 | 0.47 |
| 30 assets | Carhart | 1.24 | 0.22 | 4.7 | 33.9 | 2 | 0.48 |
| 30 assets | FF5 | 1.29 | 0.18 | 4.7 | 33.6 | 2 | 0.52 |
| FF 25 size × B/M | CAPM | 1.32 | 0.17 | 3.5 | 7.8 | 0 | 0.69 |
| FF 25 size × B/M | FF3 | 1.22 | 0.24 | 1.7 | 7.7 | 1 | 0.91 |
| FF 25 size × B/M | Carhart | 1.18 | 0.28 | 1.6 | 7.1 | 1 | 0.91 |
| FF 10 momentum | CAPM | 0.92 | 0.52 | 2.8 | 10.0 | 0 | 0.75 |
| FF 10 momentum | FF3 | 0.74 | 0.68 | 2.4 | 5.6 | 0 | 0.79 |
| FF 10 momentum | Carhart | 1.03 | 0.42 | 1.9 | 3.6 | 0 | 0.86 |

&#91;image: Figure 5: alphas under CAPM, FF3 and Carhart for the 30 assets\]

**Figure 5.** *NVDA's alpha (about +34% a year) survives every model; no other asset's alpha is significant under any model.* Filled markers have |t| > 2.

The loadings explain why the alphas barely move. They are large and sensible, but they multiply premia that were close to zero.

**Table 8 — Carhart loadings, selected assets.**

| Asset | Market | SMB | HML (t) | Mom | R² |
| --- | --- | --- | --- | --- | --- |
| NVDA | 1.80 | −0.22 | −0.98 (−3.9) | 0.08 | 0.45 |
| AMZN | 1.26 | −0.17 | −0.94 (−6.5) | −0.12 | 0.59 |
| MSFT | 0.96 | −0.70 | −0.60 (−5.8) | −0.24 | 0.60 |
| XLE | 0.95 | 0.05 | 1.13 (8.3) | −0.33 | 0.62 |
| BAC | 1.29 | 0.10 | 0.95 (9.7) | −0.12 | 0.78 |
| KO | 0.51 | −0.68 | 0.33 (3.5) | −0.09 | 0.36 |

Tech names load as growth stocks (negative HML), energy and banks as value stocks, and the mega-caps as big (negative SMB). Where both loading and premium matter, alphas do shrink: KO falls from +4.2% to +1.0% and PG from +2.0% to −1.0% under FF3.

On the sorted portfolios the factors do their job. FF3 halves the mean pricing error on the 25 size × book-to-market portfolios (3.5% to 1.7% a year). The small-growth corner goes from −7.8% to −0.4%. On the momentum deciles, adding Mom shrinks the largest alpha from 10.0% to 3.6%, and the loser decile from −10.0% to −0.7%.

### (3c) Verdict: which Stage 2 anomalies survive?

1. **The single-stock anomaly survives.** NVDA's alpha is unchanged by FF3, Carhart and FF5 (t ≈ 2.9). It is not a hidden factor exposure; it is one stock-specific outcome, consistent with chance plus survivorship. MSFT "emerging" under Carhart (t = 1.99 to 2.01) is a knife-edge, not a finding.
2. **Portfolio-level anomalies are absorbed.** Size and value alphas halve under FF3, and Mom absorbs the momentum spread, as Fama and French (1993) and Carhart (1997) intended. The one cell that becomes significant (big growth, t = 2.2) is within the 1.2 per model expected by chance, and is the mega-cap tech story again.
3. **The flat SML is only partly a factor story.** On portfolios, FF3 removes half the CAPM's pricing errors. On our assets, factors explain variance (R² +0.11) but not means, because this decade paid nothing for size, value or momentum.

### A self-built factor: industry momentum

The handout gives credit for building a factor. Momentum needs only returns, so we built it ourselves on the 10 Ken French industries since 1964. Each month we rank industries on their return from t−12 to t−2, buy the top three and short the bottom three, equally weighted, and hold for one month.

- **It is a real factor.** It earns 3.8% a year (t = 2.2) over 746 months. Its alpha is 4.0% against the CAPM (t = 2.3) and 4.9% against FF3 (t = 2.8), so it is priced beyond the market, size and value.
- **It matches the professional version.** It correlates 0.72 with Ken French's stock-level Mom: the same phenomenon at the industry level (Moskowitz and Grinblatt 1999).
- **Trading costs can eat it.** It turns over 82% of its gross position each month. At 50 bp per unit of turnover its premium becomes −1.1% a year; it breaks even at about 38 bp. A factor can be priced and still not be a profitable strategy.

The same recipe on our 30 assets (107 months) also correlates 0.72 with Ken French's Mom but has t = 0.7, too short to test. We do not use it to revise the verdict above, because a factor sorted from the test assets explains them mechanically.

## 6. Stage 4 — Judgment

The evidence comes first, then the two-page memo it supports. Out of sample, no rule beats the value-weighted index significantly, and every CAPM verdict depends on the market proxy.

### Evidence A — Roll's critique in our data

We re-run Stage 2 against six other proxies. The handout's example, "local index versus MSCI World", is the S&P 500 (SPY) against MSCI World (URTH); Ken French's developed-markets series is a fee-free check (correlation 0.996 with URTH).

**Table 9 — The CAPM under seven market proxies.** *Changing the proxy changes how many alphas look significant (0 to 4), the GRS p-value (0.10 to 1.00) and NVDA's alpha (0 to 40%).*

| Proxy | Proxy Sharpe | Mean β | Mean \|Δβ\| | \|t(α)\| > 2 | GRS | GRS p | NVDA α (% p.a.) | SML slope (% p.a.) | Proxy premium (% p.a.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRSP US market (baseline) | 0.83 | 0.88 | — | 1 | 1.25 | 0.21 | +35.2 | 18.1 | 13.3 |
| S&P 500 (SPY) | 0.87 | 0.92 | 0.04 | 1 | 1.40 | 0.12 | +34.3 | 18.4 | 13.3 |
| MSCI World (URTH) | 0.76 | 0.95 | 0.07 | 4 | 1.43 | 0.10 | +38.3 | 17.9 | 11.3 |
| Developed markets (Ken French) | 0.73 | 0.93 | 0.06 | 4 | 1.31 | 0.17 | +39.6 | 17.3 | 10.9 |
| Our universe, value-weighted | 1.08 | 0.76 | 0.16 | 1 | 1.23 | 0.23 | +23.7 | 19.4 | 18.5 |
| Our universe, 1/N | 1.04 | 1.00 | 0.12 | 3 | 1.05 | 0.41 | +31.5 | 17.9 | 15.1 |
| Ex-post tangency of the 30 assets | 2.40 | 0.23 | 0.64 | 0 | 0.00 | 1.00 | 0.0 | 64.9 | 64.9 |

&#91;image: Figure 6: CAPM alphas under four market proxies\]

**Figure 6.** *Against MSCI World four alphas are significant (AAPL, MSFT, NVDA, XLK) instead of one; the ranking of alphas barely changes.*

GRS depends only on Sharpe ratios. It measures how far the proxy sits from the efficient frontier of the assets plus the proxy: SR\_max is the best Sharpe ratio available from them (2.4–2.6 here), SR\_p the proxy's own. With the in-sample tangency portfolio as the proxy the gap is zero, every alpha is exactly zero and the SML fits perfectly.

```latex
\mathrm{GRS} = \frac{T-N-1}{N}\,\frac{SR_{\max}^2 - SR_p^2}{1 + SR_p^2}
```

### Evidence B — An out-of-sample horse race

Each month every rule sees only the previous 60 months, forms its weights, and earns the next month's return. We charge 50 bp per unit of turnover and test each Sharpe ratio against the value-weighted portfolio (Jobson–Korkie with the Memmel correction). A 36-month window is not usable with 30 assets: in 32 of 84 such windows the plug-in tangency portfolio falls on the inefficient side of the frontier.

**Table 10 — Our 30 assets, Sep 2021 – Aug 2026 (59 months).** *The plug-in tangency portfolio is wiped out; the sensible rules are indistinguishable from the index.*

| Rule | Excess return (% p.a.) | Vol (% p.a.) | Sharpe | p vs index | Sharpe net of costs | CEQ, γ = 1 (% p.a.) | Turnover (% per month) | Gross leverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Value-weighted index | 15.2 | 18.5 | 0.82 | — | 0.81 | 13.5 | 3 | 1.0× |
| 1/N | 13.1 | 14.6 | 0.90 | 0.76 | 0.88 | 12.0 | 5 | 1.0× |
| Minimum variance | 14.1 | 17.6 | 0.80 | 0.96 | 0.36 | 12.5 | 131 | 7.5× |
| Tangency, plug-in | −32.2 | 131.7 | −0.24 | 0.12 | −0.67 | −118.8 | 12,935 | 44.1× |
| Tangency, long-only | 17.1 | 17.9 | 0.95 | 0.69 | 0.90 | 15.5 | 19 | 1.0× |
| Tangency, shrinkage | 9.2 | 40.1 | 0.23 | 0.26 | −0.10 | 1.1 | 226 | 7.4× |
| Tangency, shrinkage, long-only | 16.9 | 19.0 | 0.89 | 0.81 | 0.82 | 15.1 | 21 | 1.0× |
| CRSP US index (investable) | 9.9 | 16.0 | 0.62 | 0.29 | 0.62 | 8.6 | 0 | 1.0× |

CEQ is the riskless return a mean–variance investor would accept instead of the rule. Our universe's index beat the investable CRSP index by 5.3% a year: that is the survivorship head start, which every rule built on this universe shares. So we repeat the race without it.

**Table 11 — Ten US industry portfolios, all CRSP stocks, 1968–2026.** *Over sixty years no rule beats the index significantly; unconstrained optimisation is significantly worse.*

| Rule | Sharpe, 60-month window | p vs index | Sharpe, 120-month window | p vs index | Net Sharpe, 120-month | Turnover (% per month) |
| --- | --- | --- | --- | --- | --- | --- |
| Value-weighted index | 0.46 | — | 0.51 | — | 0.51 | 0.5 |
| 1/N | 0.50 | 0.14 | 0.55 | 0.16 | 0.54 | 2.4 |
| Minimum variance | 0.50 | 0.74 | 0.57 | 0.58 | 0.50 | 15.9 |
| Tangency, plug-in | −0.21 | < 0.001 | −0.14 | < 0.001 | −0.22 | 931 |
| Tangency, long-only | 0.46 | 0.98 | 0.45 | 0.47 | 0.40 | 15.3 |
| Tangency, shrinkage | −0.14 | 0.001 | 0.10 | 0.009 | 0.02 | 49.7 |
| Tangency, shrinkage, long-only | 0.47 | 0.80 | 0.47 | 0.45 | 0.42 | 13.5 |
| CRSP US index | 0.45 | 0.02 | 0.51 | 0.13 | 0.51 | 0 |

The 60-month window starts in July 1968 (698 months), the 120-month window in July 1973 (638 months). The CRSP row's p = 0.02 is an artefact: the two indices correlate 0.9996, so a 0.01 Sharpe gap looks "significant".

&#91;image: Figure 7: out-of-sample growth of $1, ten US industries, 1973–2026\]

**Figure 7.** *Over 1973–2026, $1 in the index grows to 37 (excess return); the long-only optimised portfolios end at 27 and minimum variance at 30, and only 1/N ends higher (44).* None of the Sharpe differences is significant; the plug-in and unconstrained shrinkage rules are not shown because they lose everything.

### Memo to the investment committee

**To:** Investment Committee · **From:** \[group\] · **Re:** Should we believe the textbook?

**Recommendation.** Hold the value-weighted market index, and set risk through the split between the index and T-bills, not by changing what is in the risky portfolio. This is the Mutual-Fund Theorem: every mean–variance investor holds the same risky fund and differs only in how much of it. With the long-run US premium (8.3% ± 1.8% a year since 1926) and volatility (18.4%), the share in the index is y\* = premium / (γσ²):

| Risk aversion γ | Share in the index | Range if the premium is ± 2 s.e. |
| --- | --- | --- |
| 2 | 124% | 69–178% |
| 3 | 82% | 46–119% |
| 5 | 49% | 28–71% |
| 10 | 25% | 14–36% |

Do not hold the estimated tangency portfolio. If the committee wants to deviate from the index, keep the deviation small, long-only and shrunk toward the market, and treat it as a risk choice rather than a source of return.

**Why not the tangency portfolio.** In theory it is the best risky portfolio, and under the CAPM it is the market. In practice it must be estimated, and the estimate is dominated by error in expected returns. On our assets it puts up to 272% of capital in one name, and its weights from two halves of the same decade correlate only 0.35. Formed each month from the previous five years, it earned a Sharpe ratio of −0.24 on our assets since 2021 and lost everything. Over sixty years of industry data it is significantly worse than the index (p < 0.001). Its in-sample Sharpe ratio of 2.4 is what fitting noise looks like.

**Why not something in between.** We tested the tangency portfolio with no short sales, with expected returns shrunk about half-way toward the CAPM prediction, and with both. Shrinkage alone is not enough: over sixty years it is significantly worse than the index. The long-only versions are safe: they match the index's Sharpe ratio (0.45–0.47 vs 0.46–0.51) but trail it after costs (0.36–0.42 vs 0.46–0.51), because they turn over 13–26% of the portfolio a month against 0.5% for the index. Two design choices were made after a first run; Appendix A2 discloses them.

The deeper reason is in Stages 2 and 3: there is no reliable expected-return signal to tilt toward. The alphas are jointly insignificant, the only significant one belongs to a single ex-post winner, and size, value and momentum earned nothing this decade. Equal weighting and minimum variance had slightly higher Sharpe ratios over sixty years, but not significantly (p ≥ 0.14), and their edge comes from tilts toward smaller or calmer industries.

**Roll's critique.** The S&P 500 changes almost nothing. MSCI World raises betas by 0.07, makes four alphas significant instead of one, and lowers the GRS p-value from 0.21 to 0.10. Our universe's own index cuts NVDA's alpha from 35% to 24% and makes the fitted SML match theory. With the in-sample tangency portfolio as the proxy, every alpha is zero. A test of the CAPM is therefore a test of whether the proxy is efficient. The true market, including bonds, property and human capital, is unobservable, so our verdicts are conditional on the proxy. The recommendation does not depend on them: the index wins on cost, diversification and robustness to estimation error, not because the CAPM is proven.

**The result we trust least.** We trust least the ranking of the sensible portfolios on our own 30 assets. It rests on 59 months of a tech-led bull market, in a universe of today's largest companies backfilled ten years. Our universe's index earned an excess return of 15.2% a year against 9.9% for the investable CRSP index, a head start no investor could have known in 2021. The ranking there (long-only tangency 0.95, 1/N 0.90, index 0.82) is not significant (p ≥ 0.69) and could reverse in the next five years. That is why the recommendation leans on the sixty-year, survivorship-free test. A close second is NVDA's alpha: one ex-post selected stock, with t = 2.9 among 30 tests (Bonferroni p = 0.14).

## 7. Appendix

### A1. Tangency portfolio with and without the sector ETFs

*The ETF overlap amplifies the extreme weights but does not cause them.*

| Universe | Assets | Condition number (correlation) | Max weight | Gross exposure | Shorts | Half-sample corr | Half-sample L1 distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All assets | 30 | 490 | 272% | 15.9× | 13 | 0.35 | 33.7 |
| Stocks only | 21 | 94 | 79% | 4.5× | 6 | 0.14 | 9.9 |
| Sector ETFs only | 9 | 54 | 111% | 3.8× | 4 | −0.02 | 7.3 |

### A2. Disclosure: choices made after a first run

Two choices in the out-of-sample race were made after seeing a first run. Both are reported.

1. The first shrinkage rule took the CAPM prior's market premium from the five-year estimation window. After bear markets that premium turns negative, which makes the prior short the market and the weights explode. An equilibrium premium must be positive, so the rule now uses the historical US premium known at each date (expanding mean since 1926).
2. When the unconstrained shrinkage rule still failed, we added the long-only shrinkage rule.

The windows (60 and 120 months), the 50 bp cost, the other rules and the test were not changed. A timing fix was also made: Ken French's industry market caps are measured at the end of the previous month, so they are used without an extra lag. Unlagged, they reproduce the CRSP market with a 13 bp monthly tracking error, against 15.5 bp when lagged.

### A3. Joint test on the 21 stocks alone

Without the ETFs, GRS = 1.58 with F(21, 97), p = 0.07; one alpha has |t| > 2 against 1.0 expected by chance. The ETFs, whose alphas are small and precisely estimated, dilute the joint test.

### A4. Self-built industry momentum factor

| Statistic | Industry momentum (10 industries) | Universe momentum (30 assets) |
| --- | --- | --- |
| First month | 1964-07 | 2017-10 |
| Months | 746 | 107 |
| Mean (% p.a.) | 3.8 (t = 2.18) | 3.7 (t = 0.67) |
| Sharpe | 0.28 | 0.22 |
| CAPM alpha (% p.a.) | 4.0 (t = 2.26) | 4.9 (t = 0.87) |
| FF3 alpha (% p.a.) | 4.9 (t = 2.80) | 3.9 (t = 0.68) |
| Correlation with Ken French Mom | 0.72 | 0.72 |
| Turnover (% per month) | 82 | 72 |
| Mean net of 50 bp (% p.a.) | −1.1 | −0.6 |
| Break-even cost (bp) | 38 | 43 |

### A5. Out-of-sample growth on our 30 assets

&#91;image: Figure A1: out-of-sample growth of $1, 30 assets, 2021–2026\]

**Figure A1.** *On our 30 assets since 2021 the sensible rules move together and all finish above the investable CRSP index, the survivorship head start.* The plug-in tangency portfolio is not shown because it loses everything.

### A6. Survivorship bias, stage by stage

- **Stage 1:** average returns of the names that won are biased upward, so the tangency portfolio loads on them and its in-sample Sharpe ratio is overstated.
- **Stage 2:** high-beta winners such as NVDA push the SML up and to the right, and positive alphas are overstated. Survivorship-free Ken French portfolios give the flatter SML.
- **Stage 3:** alphas that come from selection are not factor exposures, so factors cannot absorb them; NVDA survives every model.
- **Stage 4:** our universe's index beat the investable CRSP index by 5.3% a year out of sample, and every rule built on the universe inherits that head start. The sixty-year industry test has no such bias.

### A7. Methods and references

- Time-series regressions by OLS, with Newey and West (1987) standard errors (4 lags).
- Joint test: Gibbons, Ross and Shanken (1989).
- SML: Fama and MacBeth (1973) with the Shanken (1992) errors-in-variables correction.
- Shrinkage: Ledoit and Wolf (2004) constant-correlation covariance; means shrunk toward the CAPM prior (Black and Litterman 1992) with a James–Stein weight (Jorion 1986); constraints as shrinkage (Jagannathan and Ma 2003).
- Out-of-sample design: DeMiguel, Garlappi and Uppal (2009); Sharpe-ratio test of Jobson and Korkie (1981) with Memmel (2003).
- Theory and evidence: Tobin (1958); Black (1972); Roll (1977); Fama and French (1993); Carhart (1997); Moskowitz and Grinblatt (1999); Frazzini and Pedersen (2014).
- Data: Kenneth French Data Library (release 2026-08); Yahoo Finance via yfinance.

### A8. Reproducibility

One command regenerates every exhibit from the committed CSV files, offline, in under ten seconds:

```bash
uv sync --extra notebook
uv run python run_all.py
uv run pytest          # 107 tests, 98% coverage
```

The entry point is `notebooks/portfolio_analysis.ipynb`; `run_all.py` executes it top to bottom. Package versions: Python 3.13.11, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2 (all pinned in `uv.lock`). The README lists every data file.

### A9. AI acknowledgement (template)

"\[Tool\] was used for \[writing and testing Python code / drafting notes for the report\]; all numbers were produced by our code and checked by the group."

### A10. Contribution statement (template, signed by all members)

| Member | Contribution | Signature |
| --- | --- | --- |
|  | Data collection and cleaning |  |
|  | Stage 1 analysis and write-up |  |
|  | Stages 2–3 analysis and write-up |  |
|  | Stage 4 memo and executive summary |  |
|  | Editing and reproducibility check |  |
