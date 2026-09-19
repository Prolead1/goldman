# FE 5108 Portfolio Theory and Investments
## Midterm Group Project

**From the Mutual-Fund Theorem to the Security Market Line: does the theory survive real data?**

*Risk Management Institute · National University of Singapore · 2026 Teaching Edition*

---

### At a glance

- **Groups:** 4–6, self-formed
- **Deadline:** Sunday 18 October 2026, 23:59 (Singapore time)
- **Submission:** One per group via email to sunyifei.econ@gmail.com
- **Deliverables:** A written report of at most 12 pages, reproducible code, and a signed contribution statement
- **No presentation**

---

## 1. Why this project

In Weeks 1–6 you built a complete chain of theory: expected utility, mean–variance portfolio choice, the Mutual-Fund Theorem, the CAPM, and factor models. Every link in that chain is a claim about data. This project asks your group to walk the whole chain on a real market and report, honestly, where the theory holds up and where it bends.

You will act as a small quantitative research team asked by an investment committee: "Should we believe the textbook? Show us, on a market of your choosing." The committee is numerate but busy: it rewards clear findings, honest caveats, and judgment — not page count.

The project is released in Week 5 and everything it needs is covered by the end of Week 6; the deadline is deliberately later, so plan your own schedule (a suggested one is in Section 5). After Week 6, questions go by e-mail.

---

## 2. The task (Track A — default)

Choose an investment universe of 20–30 risky assets with at least 8 years of monthly returns (e.g., Straits Times Index constituents, S&P 500 sector ETFs, a non-US country's large caps, or a diversified ETF menu). Choose a riskless-rate proxy and a market proxy, and justify both choices. Then work through four stages. Each stage names its required exhibits; these form the skeleton of your report.

### Stage 1 · Portfolio choice (Weeks 2–3)

Estimate the mean vector μ̂ and covariance matrix Σ̂ from your sample. Compute the tangency portfolio **w_F ∝ Σ̂⁻¹μ̂** and compare it with the value-weighted portfolio of the same universe.

**Required exhibits:**
- **(1a)** Summary statistics of your universe
- **(1b)** The tangency weights next to the value weights, as one chart
- **(1c)** A short discussion: the two will disagree sharply — typically with extreme long–short tangency positions — so what does that tell you about estimation error in μ̂ and Σ̂? (Try re-estimating on a subsample and watch the weights move.)

### Stage 2 · The CAPM (Week 4)

For each asset, run the security characteristic regression:

**R_i,t − R_f,t = α_i + β_i(R_M,t − R_f,t) + ε_i,t**

Plot the empirical SML: average excess return against β̂_i, with the theoretical line R̄_M − R̄_f drawn through the origin of that space.

**Required exhibits:**
- **(2a)** A table of β̂_i, α̂_i, t(α̂_i), and R²
- **(2b)** The empirical SML scatter with fitted and theoretical lines
- **(2c)** A joint assessment of "all alphas are zero" — a formal GRS test is welcome but not required; a careful discussion of how many |t(α̂)| > 2 you would expect by chance is acceptable
- **(2d)** The fitted SML's slope and intercept versus theory (too flat? intercept too high?), with interpretation

### Stage 3 · Factors (Weeks 5–6)

Augment the market model with at least two additional factors (size, value, momentum — from Ken French's library for US universes, or self-built long–short spread portfolios for other markets; building your own earns credit). Re-examine the assets or, better, sorted portfolios.

**Required exhibits:**
- **(3a)** Factor construction description (one paragraph each)
- **(3b)** Alphas before versus after adding factors, as one comparison table or chart
- **(3c)** A verdict: which Stage-2 anomalies are absorbed by factors, and which survive?

### Stage 4 · Judgment (the part that separates groups)

Write the memo to the investment committee: at most two pages inside the report. Address explicitly:

- What would you tell an investor to do — hold the value-weighted index, the estimated tangency portfolio, or something in between — and why?
- Roll's critique in your own data: redo Stage 2 with a different market proxy (e.g., local index versus MSCI World). How much do betas and alphas move? What does that imply about "testing the CAPM"?
- One paragraph of intellectual honesty: the result in your project you trust least, and why.

---

## 3. Alternate tracks (optional, same deliverables and rubric)

Groups preferring a different emphasis may choose one of the following; e-mail the instructor for approval by the Week 6 session.

**Track B — The horse race.** Replicate, on your own universe, the spirit of DeMiguel, Garlappi and Uppal (2009): compare out-of-sample (rolling-window) performance of 1/N, the value-weighted index, minimum variance, and the estimated tangency portfolio, with and without simple fixes (shrinkage, no-short-sale constraints). Explain why naive rules are hard to beat, using the language of estimation error from Weeks 2–3.

**Track C — Build one factor.** Construct a single factor from scratch in a market where you cannot download it (e.g., momentum or value in Singapore or another home market): portfolio sorts, long–short spread, transaction-cost discussion, and a test of whether your factor is priced beyond the market factor. Depth on one factor beats breadth on five.

---

## 4. Data and tools

Any language is fine; Python or R expected in practice. Free data sources that suffice:

- **Yahoo Finance (yfinance)** for prices
- **Ken French Data Library** for US factors and riskless rates
- **MAS or FRED** for interest-rate proxies

Download once and cache the raw CSVs — Yahoo rate-limits repeated pulls; if it throttles you, Stooq (via pandas-datareader) is a free fallback, and a backup price bundle with a starter notebook is posted on Canvas.

Document every cleaning decision (delistings, dividends/total-return adjustment, currency). If your universe has survivorship bias — e.g., today's index members backfilled ten years — say so and discuss the direction of the bias; acknowledging it is fine, hiding it is not.

---

## 5. Deliverables and schedule

1. **Report** (due 18 October, max 12 pages including exhibits, plus an appendix of any length). Structured by the four stages, with a one-page executive summary up front for the committee. Every figure needs a takeaway sentence in its caption. PDF only.
2. **Code.** A zip or repository link that reproduces every exhibit from raw data with one entry point (run_all.py or a notebook executed top to bottom), plus a README naming the data files and package versions. We will spot-run it.
3. **Contribution statement** (half page, inside the appendix). Who did what, signed by all members. Contributions are expected to be roughly equal; large imbalances can move individual grades.

**Suggested schedule.**
- **Weeks 5–6:** Form the group, choose the universe, download and clean the data, finish Stage 1 (the theory is all in hand by Week 6)
- **By early October:** Stage 2 done
- **Week of 5–11 October:** Stage 3 and robustness checks
- **Final week:** Stage 4 memo, editing, and a full rerun of the code from scratch

Groups that leave data cleaning to October do not finish.

---

## 6. Grading rubric

| Component | Weight | What earns the marks |
|---|---|---|
| Correctness of methods | 30% | Estimators, regressions and portfolio math correctly implemented and correctly interpreted; units and annualization consistent. |
| Economic reasoning | 30% | Results connected to the theory of Weeks 1–6; Stage 4 memo shows judgment, not hedging; limitations understood (estimation error, proxy problem, survivorship). |
| Data craft & reproducibility | 20% | Clean, documented pipeline; code reruns; robustness check(s) beyond the minimum. |
| Communication | 15% | Report readable by the "busy committee"; executive summary states the findings, not the process; exhibits self-contained. |
| Teamwork | 5% | Contribution statement credible and signed. |

A strong project is not the one where the CAPM works. Most likely your SML will be too flat and some alphas will survive — that is the finding of fifty years of literature. Marks follow the quality of the investigation, not the verdict.

---

## 7. Frequently asked questions

**Which risk-free rate?** A short-term government bill or the policy rate in the currency of your universe (US: the one-month T-bill from Ken French's library; Singapore: MAS bills or SORA). State it once; monthly rate = annual/12 is fine.

**Monthly or weekly returns?** Monthly, at least 96 observations. Weekly data give more points but noisier betas and microstructure effects; use them only as a robustness check.

**Our universe is today's index members. Is that a problem?** Yes — survivorship bias: firms that failed are missing, so average returns are biased upward and the SML may look better than it should. Acknowledge it, say which way it biases each stage, and if you can, use historical constituents or include delisted names.

**The tangency weights are absurd (e.g., +340% in one stock). Did we make a mistake?** Probably not. That is estimation error in μ̂ and Σ̂ doing exactly what Weeks 2–3 warned about. Report it, explain it, and show how a no-short constraint or shrinkage tames it. This is a finding, not a bug.

**Our alphas are not zero. Does the CAPM "fail"?** First ask whether they are jointly significant — with 25 assets you expect one or two |t| > 2 by chance. Then ask whether factors absorb them (Stage 3), and whether a different market proxy changes them (Stage 4). Only then write the verdict.

**Can we use the same universe as another group?** Yes, but your analysis and text must be your own. Different universes across groups make for a better class discussion, so consider a market you know.

**How do we acknowledge AI tools?** One sentence in the appendix: which tools, for what (e.g., "ChatGPT for debugging pandas code and copy-editing Section 4"). Do not cite an AI tool as a source for a fact or a number.

**We are unsure whether a robustness check counts.** If it changes a conclusion or shows that a conclusion is stable, it counts. If it only adds a table, it does not.

---

## 8. Ground rules

**Collaboration.** Work within your group; discussion across groups is fine, sharing code or text is not.

**AI tools.** Permitted for coding assistance, editing, and exploration, and must be acknowledged in one sentence in the appendix (which tools, for what). You are responsible for every number and claim: "the model said so" is not a defence, and fabricated citations or unverifiable numbers are treated as misconduct.

**Sources.** Cite data sources and any papers you lean on. Suggested background: the Week 2–6 slides; BKM (Investments) Chs. 7–10 and 13; Fama and French (1993); DeMiguel, Garlappi and Uppal (2009); Roll (1977); Gibbons, Ross and Shanken (1989) for the joint alpha test.

**Advice from the instructor.** Start with the data this week — every late project died of data cleaning, none died of regressions. Keep the universe modest: 25 clean assets beat 100 messy ones. And write Stage 4 first as a guess, then let the analysis correct you; that is what research feels like.
