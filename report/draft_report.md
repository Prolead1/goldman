# FE 5108 report kit — draft text for the PDF

> **Draft for the group to rewrite.** Every number below comes from
> `notebooks/portfolio_analysis.ipynb` run on the committed data (US universe,
> Oct 2016 – Aug 2026). The handout requires the analysis and text to be the
> group's own and AI use to be acknowledged in the appendix: treat this file as
> notes to check, cut and rewrite, not as finished prose. Re-run the notebook and
> re-check every figure before submitting.

**Contents**

1. Exhibit plan for the 12-page limit (main text vs appendix, with captions)
2. Executive summary (one page)
3. Stage 4 memo to the investment committee (two pages)
4. Appendix notes: data and cleaning, survivorship by stage, AI acknowledgement,
   contribution statement

---

## 1. Exhibit plan and captions

The notebook writes about 55 files to `outputs/`. The report has room for about
ten exhibits. Every caption needs a takeaway sentence; suggested ones follow.

### Main text

| # | Stage | File | Caption (takeaway first) |
|---|---|---|---|
| T1 | 1a | `summary_stats.csv` (condensed: mean, vol, Sharpe) | *Ten years of mega-cap returns:* average returns of 7–61% p.a., volatilities of 13–53%, Sharpe ratios from 0.15 (DIS) to 1.25 (NVDA). |
| F1 | 1b | `tangency_weights.png` | *The estimated tangency portfolio bears no resemblance to the market:* positions up to 272% of capital, long and short, against value weights below 16%. |
| F2 | 1c | `taming_weights.png` (+ `taming_table.csv` as a small table) | *A no-short constraint or shrinkage cuts the largest position from 272% to 17–46%, but the weights estimated on the two halves of the sample remain uncorrelated:* taming limits the damage of estimation error without removing it. |
| T2 | 2a | `capm_table.csv` (β, α, t(α), R²) | *Betas are precise, alphas are not:* every t(β) > 2, but only NVDA's alpha has \|t\| > 2. |
| F3 | 2b/2d | `sml.png` | *Our 30-asset SML looks steeper than theory only because of NVDA;* without it, and on the Ken French portfolios (`ff25_sml.png`), the SML is flatter than theory with a positive intercept. |
| T3 | 2c/2d | `alpha_joint_test.csv` + `sml_robustness.csv` | *We cannot reject that all alphas are zero (GRS p = 0.21; 0.07 on the 21 stocks alone), and with ten years of data an alpha has to exceed about 16% p.a. to be detectable; no SML slope or intercept differs significantly from theory in any sample.* |
| T4 | 3a/3b | `factor_summary.csv` + `model_summary.csv` + `sorted_portfolios_model_summary.csv` | *Only the market earned a premium in 2016–2026,* so factors raise R² (0.37 → 0.48) without moving our alphas; on sorted portfolios FF3 halves the size/value alphas and momentum absorbs the loser decile. |
| F4 | 3b/3c | `alpha_comparison.png` | *NVDA's alpha survives every factor model;* no other asset's alpha is significant under any model. |
| F5 | 4a | `roll_alphas.png` (+ `roll_summary.csv` rows: GRS p, mean \|Δβ\|, \|t\|>2, SML slope) | *Switching from the US market to MSCI World raises the number of significant alphas from one to four and pushes the GRS p-value from 0.21 to 0.10;* with the in-sample tangency portfolio as proxy, every alpha is exactly zero. |
| T5 | 4b/4c | `horse_race_industries_1963.csv` (Sharpe, p vs VW, net Sharpe, CEQ, drawdown) + `horse_race_us30.csv` | *Out of sample no rule beats the value-weighted index significantly, in either data set; over sixty years unconstrained optimisation is significantly worse (p ≤ 0.01) and wipes the investor out.* |
| F6 | 4c | `horse_race_industries_1963.png` | *Over 1973–2026 $1 in the index grows to 37 (excess return); the long-only optimised portfolios end at 27 and minimum variance at 30, and only 1/N ends higher (44) — none of the Sharpe differences is significant.* |

The two-fund allocation table goes inside the memo (section 3), not as a separate
exhibit.

### Appendix

| # | File | Caption |
|---|---|---|
| A1 | `tangency_by_subset.csv` | *The ETF/constituent overlap amplifies the extreme weights but does not cause them:* without the ETFs the condition number falls from 490 to 94 and the largest weight from 272% to 79%, yet the two half-sample portfolios still correlate only 0.14. |
| A2 | `alpha_joint_test_stocks.csv` | *On the 21 stocks alone the alphas are jointly significant at 10% but not 5% (GRS p = 0.07).* |
| A3 | `self_built_momentum.csv` | *Our self-built industry-momentum factor is priced (FF3 alpha 4.9% p.a., t = 2.8, 1964–2026) and correlates 0.72 with Ken French's, but it breaks even at about 38 bp of trading cost.* |
| A4 | `roll_summary.csv` (all seven proxies), `roll_betas.png` | *The S&P 500 behaves like the CRSP market; MSCI World (URTH) and Ken French's developed-markets series agree with each other and both raise betas by about 0.07.* |
| A5 | `horse_race_us30.png`, `horse_race_us30.csv` | *On our 30 assets since 2021 the sensible rules are indistinguishable (p ≥ 0.69) and all beat the investable CRSP index — the survivorship head start.* |
| A6 | `two_fund_allocation.csv` | *Index share by risk aversion, with the premium at ± 2 standard errors.* |

Also for the appendix if space allows: `subsample_weights.csv`, `ff25_sml.png`,
`sml_tests.csv`, the full `alpha_comparison.csv`, `factor_loadings.csv`,
`anomaly_verdict.csv` and the `ff25_`, `mom10_`, `ff5_` variants.

---

## 2. Executive summary

**Question.** Does the textbook chain — mean–variance choice, the Mutual-Fund
Theorem, the CAPM and factor models — survive ten years of monthly data on 30
large US assets (21 mega-cap stocks, 9 sector ETFs)? Market proxy: the CRSP
value-weighted index; risk-free rate: the one-month T-bill.

**1. The estimated optimal portfolio is mostly noise.** The tangency portfolio
holds positions up to 272% of capital; re-estimated on each half of the sample,
its weights correlate only 0.35. A no-short constraint or shrinkage toward the
CAPM cuts positions to 17–46% but does not make the two halves agree.

**2. The CAPM is not rejected, mostly because ten years cannot reject it.** One
alpha (NVDA, +35% p.a.) has |t| > 2, against 1.4 expected by chance; GRS
p = 0.21 (0.07 on the stocks alone). An alpha must exceed about 16% p.a. to be
detectable. Our SML is steeper than theory only because of NVDA; without it, and
on survivorship-free Ken French portfolios, it is flatter than theory with a
positive intercept, as the literature finds — though never significantly so.

**3. Factors explain risk, not return, in this decade.** Size, value and momentum
earned roughly nothing in 2016–2026, so they raise R² from 0.37 to 0.48 without
moving alphas; on sorted portfolios they do their job. NVDA's alpha survives every
model. Our own industry-momentum factor is priced over 1964–2026 (FF3 alpha 4.9%,
t = 2.8) but costs about as much to trade as it earns.

**4. Every CAPM verdict depends on the proxy.** Against MSCI World four alphas are
significant instead of one; against the in-sample tangency portfolio every alpha
is exactly zero. A test of the CAPM is a test of the proxy (Roll 1977).

**Recommendation: hold the value-weighted index, mixed with T-bills according to
risk tolerance** (about 80% index for moderate risk aversion). Out of sample — on
our assets since 2021 and on ten US industries since 1963 — no rule beats the
index significantly; the plug-in tangency portfolio is significantly worse and
wipes the investor out; constrained or shrunk versions are safe but trail the
index after costs over sixty years.

**Least trusted result:** the ranking of the sensible rules on our own universe —
59 months of a tech-led bull market on a survivorship-biased list.

---

## 3. Memo to the investment committee

**To:** Investment Committee  **From:** [group]  **Re:** Should we believe the
textbook?  **Date:** [date]

**Recommendation.** Hold the value-weighted market index, and set risk with the
split between the index and T-bills, not by changing what is in the risky
portfolio. This is the Mutual-Fund Theorem: every mean–variance investor holds
the same risky fund and differs only in how much of it they hold. With the
long-run US premium (8.3% ± 1.8% a year) and volatility (18.4%):

| Risk aversion γ | Index share y* = premium / (γσ²) | Range if the premium is ± 2 s.e. |
|---|---|---|
| 2 | 124% | 69–178% |
| 3 | 82% | 46–119% |
| 5 | 49% | 28–71% |
| 10 | 25% | 14–36% |

Do not hold the estimated tangency portfolio. If the committee wants to deviate
from the index, keep the deviation small, long-only and shrunk toward the market,
and treat it as a risk choice, not a source of return.

**Why not the tangency portfolio.** In theory it is the best risky portfolio, and
under the CAPM it is the market. In practice it must be estimated, and the
estimate is dominated by error in expected returns. On our 30 assets it takes
positions of up to 272% of capital, and the weights estimated on two halves of the
same decade correlate only 0.35. We tested it the only fair way: each month we
formed it from the previous five years and held it for one month.

- **Our 30 assets, 2021–2026:** Sharpe ratio −0.24 against 0.82 for the index,
  average leverage of 44 times capital, and one month that lost more than
  everything.
- **Ten US industry portfolios, 1968–2026:** about 700 months on every CRSP stock,
  free of our selection bias. Again significantly worse than the index
  (p < 0.001).

Its in-sample Sharpe ratio of 2.4 is what fitting noise looks like.

**Why not "something in between".** We tested the tangency portfolio with no short
sales, with expected returns shrunk about half-way toward the CAPM's prediction
(Ledoit–Wolf covariance), and with both. Shrinkage alone is not enough: over sixty
years it is significantly worse than the index and loses everything in some
windows. The long-only versions are safe: over sixty years they match the index's
Sharpe ratio (0.46–0.47 vs 0.46–0.51, p ≥ 0.45) but trail it after a 50 bp trading
cost (0.36–0.42 vs 0.46–0.51), because they turn over 13–26% of the portfolio a
month against 0.5% for the index. (Two design choices were made after a first run;
the appendix discloses them.)

The deeper reason is in Stages 2–3: there is no reliable expected-return signal to
tilt toward. The alphas are jointly insignificant, the only significant one is a
single ex-post winner, and size, value and momentum earned nothing this decade.
Equal weighting and minimum variance had slightly higher Sharpe ratios over sixty
years (0.50–0.58), but not significantly (p ≥ 0.14), and their edge comes from
tilts toward smaller or calmer industries, not from knowledge of returns.

**Roll's critique in our data.** We re-ran the CAPM tests against six other
proxies. The S&P 500 changes almost nothing. MSCI World raises betas by 0.07 on
average, makes four alphas significant instead of one (AAPL, MSFT, NVDA, XLK) and
pushes the GRS p-value from 0.21 to 0.10. Our universe's own index cuts NVDA's
alpha from 35% to 24% and makes the fitted SML match theory. With the in-sample
tangency portfolio as the proxy, every alpha is exactly zero. The GRS statistic
measures only how far the proxy's Sharpe ratio falls short of the best attainable
one, so every test of the CAPM is a test of whether the proxy is efficient. The
true market — including bonds, property and human capital — is unobservable, so
our verdicts are conditional on the proxy. The recommendation does not depend on
them: the index wins on cost, diversification and robustness to estimation
error, not because the CAPM was proven.

**The result we trust least.** The ranking of the sensible portfolios on our own
30 assets. It rests on 59 months of a tech-led bull market, in a universe of
today's largest companies backfilled ten years — which is why our universe's index
earned an excess return of 15.2% a year against 9.9% for the investable CRSP index.
The ordering there (long-only tangency 0.95, 1/N 0.90, index 0.82) is not
significant (p ≥ 0.69) and could reverse in the next five years. That is why the
recommendation leans on the sixty-year, survivorship-free test. A close second is
NVDA's alpha: one ex-post selected stock, t = 2.9 among 30 tests (Bonferroni
p = 0.14).

---

## 4. Appendix notes

### Data and cleaning decisions (state each once)

- **Prices.** Yahoo Finance monthly adjusted closes (splits and dividends), used
  as a total-return proxy. 30 US assets, 2016-09 to 2026-08; simple monthly
  returns; tickers with fewer than 90 observations are dropped (none in the US
  universe); only months where every asset has a return are kept (119 months).
- **Timing.** Monthly bars are labelled with the month's first day but carry the
  month-end close, so the return labelled October is October's return; it is
  matched to Ken French's October factor values.
- **Risk-free rate and market proxy.** Ken French one-month T-bill and CRSP
  value-weighted market (`Mkt-RF`), library vintage 2026-08. Re-downloading
  revised one month of the earlier cached file by at most 4 bp.
- **Other proxies (Stage 4).** S&P 500 and MSCI World from the SPY and URTH ETFs
  (Yahoo adjusted closes, so total returns net of 0.09% / 0.24% fees); Ken French
  developed-markets factors (USD, same T-bill) as a fee-free check (correlation
  0.996 with URTH).
- **Factors and test portfolios.** Ken French SMB, HML, Mom, RMW, CMA; 25
  size × book-to-market and 10 momentum portfolios (value-weighted); 10 industry
  portfolios since 1963 with number of firms × average firm size as market cap.
  Ken French measures that size at the end of the previous month, so it is used
  directly as the weight held during the month (it reproduces the CRSP market
  with a 13 bp monthly tracking error).
- **Value weights.** Point-in-time: split-unadjusted close × shares outstanding,
  shares forward-filled from report dates. A value-weighted investor holds the
  weights from the end of the previous month. Sector ETFs enter with their fund
  size (1.4% of the index in total), and they also hold the stocks — the index
  double-counts those holdings slightly.
- **Currency.** All USD.

### Disclosure: choices made after a first run (Stage 4)

Two choices in the out-of-sample race were made after seeing a first run; both
are reported.

1. The first shrinkage rule took the CAPM prior's market premium from the
   five-year estimation window. After bear markets that premium is negative,
   which makes the prior *short* the market and the weights explode. An
   equilibrium premium must be positive, so the rule now uses the historical US
   premium known at each date (expanding mean of the Ken French market since
   1926).
2. When the unconstrained shrinkage rule still failed, we added a long-only
   shrinkage rule.

The window (60 months, plus 120 for the long sample), the 50 bp cost, the other
five rules and the test were not changed.

A third correction was a timing fix, not a design choice. We first lagged Ken
French's industry market caps by a month. The data show they are already
measured at the end of the previous month: unlagged weights reproduce the CRSP
market with a 13 bp monthly tracking error, against 15.5 bp when lagged.

### Self-built factor (Stage 3, for credit)

Industry momentum is built from the 10 Ken French industries. Each month the
assets are ranked on their return from t−12 to t−2. The portfolio goes long the
top 3 and short the bottom 3 with equal weights and holds for one month; the
first factor month is 1964-07.

| Statistic | Value |
|---|---|
| Months | 746 |
| Mean return | 3.8% a year (t = 2.2) |
| CAPM alpha | 4.0% (t = 2.3) |
| FF3 alpha | 4.9% (t = 2.8) |
| Correlation with Ken French Mom | 0.72 |
| Turnover | 82% a month |
| Net return after 50 bp per unit of turnover | −1.1% a year |
| Break-even cost | about 38 bp |

The same recipe on our 30 assets (top 10 minus bottom 10, 107 months) also
correlates 0.72 with Ken French Mom but has t = 0.7. It is reported only as an
illustration. It is not used to revise the Stage 3 verdict, because a factor
sorted from the test assets explains them mechanically.

### Statistical power (Stage 2)

With 119 months, the smallest alpha that a 5% test detects with 80% power is
about 16% a year for the median asset, computed as (1.96 + 0.84) × se(α). Failing
to reject the CAPM is therefore weak evidence for it.

### Survivorship bias, stage by stage (direction)

The universe is today's large-cap stocks and today's sector ETFs, backfilled ten
years. Firms that shrank, were acquired or failed are missing.

- **Stage 1:** average returns, and so `μ̂`, are biased upward for the names
  that won. The tangency portfolio loads on them and its in-sample Sharpe ratio
  is overstated.
- **Stage 2:** high-beta winners such as NVDA push the SML up and to the right,
  making it look steeper than it should. Positive alphas are overstated. The
  survivorship-free Ken French portfolios give the flatter SML.
- **Stage 3:** selection-driven alphas are not factor exposures, so factors
  cannot absorb them. NVDA's alpha survives every model.
- **Stage 4:** our universe's index beat the investable CRSP index by 5.3% a year
  (excess return) out of sample, and every rule built on the universe inherits that head start.
  The 60-year industry test has no such bias.

### Methods to cite

Gibbons, Ross and Shanken (1989) GRS test; Fama and MacBeth (1973) with the
Shanken (1992) correction; Newey and West (1987); Ledoit and Wolf (2004)
constant-correlation shrinkage; Jorion (1986) and Black–Litterman (1992) for
shrinking means toward an equilibrium prior; Jagannathan and Ma (2003) on
constraints as shrinkage; DeMiguel, Garlappi and Uppal (2009); Jobson and Korkie
(1981) with Memmel (2003); Roll (1977); Fama and French (1993); Carhart (1997);
Black (1972); Frazzini and Pedersen (2014); Moskowitz and Grinblatt (1999) for
industry momentum; Tobin (1958) for two-fund separation.

### AI acknowledgement (one sentence; fill in honestly)

"[Tool] was used for [e.g. writing and testing Python code and drafting notes
for the report]; all numbers were produced by our code and checked by the group."

### Contribution statement (template, half a page, signed by everyone)

| Member | Contribution | Signature |
|---|---|---|
| | Data collection and cleaning | |
| | Stage 1 analysis and write-up | |
| | Stage 2–3 analysis and write-up | |
| | Stage 4 memo and executive summary | |
| | Editing, reproducibility check | |
