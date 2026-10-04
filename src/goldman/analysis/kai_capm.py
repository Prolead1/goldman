"""Stage 2: asset-level CAPM, descriptive SML, and alpha diagnostics."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .. import config
from ..data import load_ff3, load_prices
from .kai_us21 import ASSETS, END, FACTOR_SOURCE, START

HAC_LAGS = 3


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the same fixed selection/window as Stage 1 without using cap estimates."""
    prices = load_prices(universe="us").loc[:, list(ASSETS)]
    prices.index = prices.index.to_period("M")
    dates = pd.period_range(pd.Period(START) - 1, END, freq="M")
    if not prices.index.is_unique:
        raise ValueError("Duplicate price months")
    prices = prices.reindex(dates)
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Require complete positive prices")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    factors = load_ff3().reindex(returns.index)
    if factors.isna().any().any():
        raise ValueError("Missing observed factors or RF")
    return returns, factors


def hac_covariance(x: np.ndarray, residuals: np.ndarray, lags: int = HAC_LAGS) -> np.ndarray:
    """Newey-West covariance: Bartlett weights and T/(T-k) small-sample scaling."""
    n, k = x.shape
    if not 0 <= lags < n or n <= k:
        raise ValueError("Invalid lag count or sample size")
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    bread = np.linalg.inv(x.T @ x)
    return bread @ meat @ bread * n / (n - k)


def fit_capm(
    excess: pd.DataFrame, market_excess: pd.Series, lags: int = HAC_LAGS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One OLS regression per asset: asset excess = alpha + beta * market excess."""
    if not excess.index.equals(market_excess.index) or not excess.index.is_unique:
        raise ValueError("Return and market dates must align uniquely")
    y, market = excess.to_numpy(dtype=float), market_excess.to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(market).all():
        raise ValueError("Regression inputs must be finite")
    n = len(excess)
    x = np.column_stack([np.ones(n), market])
    if n <= 2 or np.linalg.matrix_rank(x) < 2:
        raise ValueError("Need observations and a varying market series")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    sse = (residuals**2).sum(axis=0)
    total = ((y - y.mean(axis=0))**2).sum(axis=0)
    if (total <= 0).any():
        raise ValueError("Asset excess returns must vary")
    inv_xx = np.linalg.inv(x.T @ x)
    standard_errors = np.sqrt(np.diag(inv_xx)[:, None] * sse / (n - 2))
    robust = np.column_stack([
        np.sqrt(np.maximum(np.diag(hac_covariance(x, residuals[:, i], lags)), 0))
        for i in range(y.shape[1])
    ])
    table = pd.DataFrame({
        "beta": coefficients[1], "alpha_monthly": coefficients[0],
        "alpha_annual": coefficients[0] * 12,
        "alpha_se_monthly": standard_errors[0], "beta_se": standard_errors[1],
        "t_alpha": coefficients[0] / standard_errors[0],
        "alpha_se_hac_monthly": robust[0], "t_alpha_hac": coefficients[0] / robust[0],
        "r_squared": 1 - sse / total,
        "mean_excess_annual": y.mean(axis=0) * 12,
        "capm_excess_annual": coefficients[1] * market.mean() * 12,
        "n_months": n,
    }, index=excess.columns)
    table.index.name = "asset"
    table["flag_abs_t_gt_2"] = table.t_alpha.abs() > 2
    table["flag_abs_hac_t_gt_2"] = table.t_alpha_hac.abs() > 2
    return table, pd.DataFrame(residuals, index=excess.index, columns=excess.columns)


def fit_sml(table: pd.DataFrame, market_excess: pd.Series) -> dict[str, float]:
    """Unweighted cross-sectional fit; coefficients are descriptive, not pricing tests."""
    x = np.column_stack([np.ones(len(table)), table.beta.to_numpy()])
    if len(table) < 3 or np.linalg.matrix_rank(x) < 2:
        raise ValueError("SML needs at least three assets with varying betas")
    y = table.mean_excess_annual.to_numpy()
    intercept, slope = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ np.array([intercept, slope])
    return {
        "fitted_intercept_annual": float(intercept),
        "fitted_slope_annual": float(slope),
        "theoretical_intercept_annual": 0.,
        "theoretical_slope_annual": float(market_excess.mean() * 12),
        "cross_sectional_r_squared": float(1 - residuals @ residuals / ((y - y.mean())**2).sum()),
        "assets": len(table),
    }


def make_charts(table: pd.DataFrame, sml: dict, out: Path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    grid = np.linspace(0, table.beta.max() + .25, 200)
    ax.plot(grid, grid * sml["theoretical_slope_annual"], color="#d89939", lw=2,
            label="CAPM: zero intercept, observed market premium")
    ax.plot(grid, sml["fitted_intercept_annual"] + grid * sml["fitted_slope_annual"],
            color="#27647b", lw=2, label="Fitted cross-sectional SML")
    ax.scatter(table.beta, table.mean_excess_annual, s=36, color="#27647b", zorder=3)
    offsets = {"JNJ": (6, 6), "KO": (-30, 3), "INTC": (8, 6),
               "BAC": (8, -12), "AMZN": (-35, 5), "AAPL": (6, 7)}
    for i, (name, row) in enumerate(table.iterrows()):
        ax.annotate(name, (row.beta, row.mean_excess_annual),
                    xytext=offsets.get(name, (5, 5 if i % 2 else -12)),
                    textcoords="offset points", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .2})
    ax.set(xlabel="Estimated market beta", ylabel="Average annualized excess return",
           title="US21 | Empirical Security Market Line")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axhline(0, color="#888888", lw=.6)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out / "sml.png", dpi=150)
    plt.close(fig)

    ordered = table.sort_values("alpha_annual")
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    positions = np.arange(len(ordered))
    ax.errorbar(ordered.alpha_annual, positions, xerr=2 * ordered.alpha_se_monthly * 12,
                fmt="o", color="#27647b", ecolor="#9daeb5", capsize=3)
    ax.set_yticks(positions, ordered.index)
    ax.axvline(0, color="#d89939", lw=1.5)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Annualized alpha; bars show +/- 2 classical standard errors",
           title="US21 | Which returns remain unexplained by market beta?")
    fig.savefig(out / "alphas.png", dpi=150)
    plt.close(fig)


def table_markdown(table: pd.DataFrame) -> str:
    lines = ["| Asset | Beta | Monthly alpha | Annual alpha | t(alpha) | HAC t(alpha) | R² |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in table.iterrows():
        lines.append(f"| {name} | {row.beta:.3f} | {row.alpha_monthly:.2%} | "
                     f"{row.alpha_annual:.2%} | {row.t_alpha:.2f} | "
                     f"{row.t_alpha_hac:.2f} | {row.r_squared:.3f} |")
    return "\n".join(lines)


def run(output_dir: Path | None = None) -> Path:
    out = Path(output_dir) if output_dir else config.OUTPUTS_DIR / "kai_us21_stage2"
    out.mkdir(parents=True, exist_ok=True)
    returns, factors = load_inputs()
    excess, market = returns.sub(factors.RF, axis=0), factors["Mkt-RF"]
    table, residuals = fit_capm(excess, market)
    sml = fit_sml(table, market)
    # Leave-one-asset-out SML fits diagnose whether one dot drives the cross-sectional line.
    influence = pd.DataFrame({
        name: fit_sml(table.drop(index=name), market) for name in table.index
    }).T.rename_axis("excluded_asset")
    flagged = table.index[table.flag_abs_t_gt_2].tolist()
    robust_flagged = table.index[table.flag_abs_hac_t_gt_2].tolist()
    all_alphas = {"n_assets": len(table), "n_months": len(excess),
                  "classical_abs_t_gt_2": flagged, "hac_abs_t_gt_2": robust_flagged,
                  "expected_flags_approx_under_null": .05 * len(table),
                  "formal_joint_test_performed": False, "hac_lags": HAC_LAGS}
    for name, frame in {"capm_results": table, "residuals": residuals,
                        "asset_excess_returns": excess, "factors_aligned": factors,
                        "sml_leave_one_out": influence}.items():
        frame.to_csv(out / f"{name}.csv", float_format="%.12g")
    (out / "sml_summary.json").write_text(json.dumps(sml, indent=2) + "\n")
    (out / "alpha_assessment.json").write_text(json.dumps(all_alphas, indent=2) + "\n")
    make_charts(table, sml, out)
    slope = sml["fitted_slope_annual"]
    theoretical = sml["theoretical_slope_annual"]
    shape = "steeper" if slope > theoretical else "flatter"
    intercept = sml["fitted_intercept_annual"]
    position = "above" if intercept > 0 else "below"
    biggest = table.alpha_annual.abs().idxmax()
    swing = (influence.fitted_slope_annual - slope).abs().idxmax()
    count_conclusion = (
        "The classical count is close to the chance expectation, so the count alone provides "
        "little evidence against the joint zero-alpha hypothesis. This does not establish "
        "that every alpha is zero."
        if len(flagged) <= 2 else
        "The classical count exceeds the rough chance expectation and motivates further "
        "joint testing, but it is not itself a calibrated joint rejection."
    )
    influence_conclusion = (
        "The fitted slope can fall on either side of the theoretical slope after removing "
        "one stock. The steeper/flatter conclusion is therefore sensitive to asset selection."
        if influence.fitted_slope_annual.min() < theoretical < influence.fitted_slope_annual.max()
        else "These single-stock exclusions preserve which side of theory the fitted slope lies on."
    )
    intel_april_return = returns.loc[pd.Period("2026-04", freq="M"), "INTC"]
    report = f"""# Kai US21 - Stage 2: CAPM

## Findings for the group meeting

The analysis uses the same **21 stocks and {len(excess)} monthly observations** as Stage 1:
**{START} through {END}**. It runs a separate regression for each stock, not a regression of
the fitted tangency portfolio. Each dependent variable is the asset return minus RF;
the explanatory variable is the supplied **Mkt-RF** series, with an intercept.

- **{len(flagged)} of 21** assets have classical |t(alpha)| > 2: {', '.join(flagged) or 'none'}.
- With Newey-West standard errors ({HAC_LAGS} monthly lags), **{len(robust_flagged)} of 21**
  exceed the same threshold: {', '.join(robust_flagged) or 'none'}.
- The fitted SML has annualized slope **{slope:.2%}**, versus the CAPM reference
  **{theoretical:.2%}**, and intercept **{intercept:.2%}**, versus zero.
- The largest absolute annualized alpha is **{biggest}: {table.loc[biggest, 'alpha_annual']:+.2%}**.
  This is a historical regression estimate, not an expected trading profit.

## Exhibit 2a - Regression results

{table_markdown(table)}

Alpha and its standard errors are estimated monthly. Annual alpha is monthly alpha multiplied
by 12, not a compounded return. Beta and t-statistics are not annualized. R² measures the share
of variation in an asset's excess returns explained by this market regression.
Classical t-statistics use residual variance SSE/(T-2). The additional HAC column uses
Bartlett weights with {HAC_LAGS} lags and the T/(T-2) small-sample correction.

## Exhibit 2b - Empirical Security Market Line

![Empirical SML](sml.png)

**Takeaway:** In this selected sample the fitted line is {shape} than the CAPM reference,
with an intercept {position} zero. Individual stocks also deviate from the theoretical line.
The fitted line is ordinary, equally weighted cross-sectional OLS of mean excess returns
on estimated betas. The theoretical line is y = beta × mean(Mkt-RF), drawn through zero.
Both vertical-axis returns and line coefficients use arithmetic annualization (×12).

## Exhibit 2c - Are all alphas plausibly zero?

Under correctly calibrated individual tests, |t| > 2 is approximately a 5% two-sided
threshold. Even when every true alpha is zero, around **21 × 5% = 1.05** flags are expected
on average. The observed classical count is **{len(flagged)}**, and the HAC count is
**{len(robust_flagged)}**. Compare these with roughly one expected chance flag, rather than
treating each individual flag as a separate proof of CAPM failure.

**Assessment:** {count_conclusion}

That expected count does not require independent assets, but a binomial probability for
the observed count would: residuals are correlated across these stocks. No such independence
assumption is made here. **This is the handout's permitted count-based assessment, not a formal
GRS test or a reported global p-value.** Classical errors assume constant residual variance
and no serial correlation; HAC relaxes these assumptions asymptotically but does not solve
multiple testing, sample selection, market-proxy error or data-quality problems.

![Alpha diagnostics](alphas.png)

**Takeaway:** Bars show annualized alpha plus/minus two classical standard errors, an
approximate individual uncertainty band. A bar excluding zero corresponds to |t(alpha)| > 2;
it is not a simultaneous confidence band across the 21 stocks. Use the HAC column as a check.

## Exhibit 2d - Fitted slope and intercept versus theory

| Annualized quantity | Empirical fit | CAPM reference |
|---|---:|---:|
| Intercept | {intercept:.2%} | 0.00% |
| Slope per unit beta | {slope:.2%} | {theoretical:.2%} |

The slope difference is **{(slope-theoretical)*100:+.2f} percentage points per unit of beta**.
The empirical relation is therefore {shape}, not necessarily the flatter SML often discussed
in textbooks. We report the observed direction rather than assuming a particular anomaly.
The intercept lies {position} zero; it describes an extrapolated fitted return at beta zero,
not a traded risk-free asset. Cross-sectional R² is **{sml['cross_sectional_r_squared']:.3f}**.

Estimated betas are noisy regressors, and stock returns are cross-correlated. Accordingly,
these line coefficients are descriptive; naive second-pass OLS significance is not reported.
Leaving out one stock at a time gives slopes from **{influence.fitted_slope_annual.min():.2%}**
to **{influence.fitted_slope_annual.max():.2%}**. Omitting **{swing}** changes the slope most,
to **{influence.loc[swing, 'fitted_slope_annual']:.2%}**. This sensitivity changes the set of
cross-sectional points only; it keeps the other stocks' original regressions and dates.
All 21 remain in the primary analysis. Full diagnostics are in `sml_leave_one_out.csv`.

**Sensitivity verdict:** {influence_conclusion}

## Interpretation and remaining work

The observed slope/intercept deviations and alpha diagnostics describe where this selected
sample departs from the CAPM reference. They do not by themselves establish rejection of
the joint zero-alpha hypothesis or of every possible market proxy. Stage 3 should check
whether size and value factors absorb the residual returns; Stage 4 should repeat the CAPM
with an alternative market proxy before the group writes its final verdict.

The flagged **Intel {intel_april_return:+.1%} return in April 2026** remains in the dataset; it has
not been independently verified. The leave-Intel-out SML slope is
**{influence.loc['INTC', 'fitted_slope_annual']:.2%}**. Removing a stock from this diagnostic
does not validate its data. Results are preliminary pending that check and group review.
The universe also carries survivorship/selection bias and is not the whole US market.

## Data and reproduction

- Inputs are cached adjusted prices and the supplied FF3 file; no downloads or fills.
- Market proxy: Fama-French US market; Mkt-RF is already an excess return. RF is the
  supplied one-month US Treasury-bill proxy. [Factor definitions]({FACTOR_SOURCE}).
- The local factor file ends in July 2026. Keeping the Stage 1 window preserves comparison
  consistency even if newer online factor releases are available.
- HAC convention: [Newey-West covariance documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).
  The project implements the matrix formula with NumPy and does not require statsmodels.
- Run `python run_kai_stage2.py`, or `notebooks/kai_stage2.ipynb` top to bottom.
  Outputs go to `outputs/kai_us21_stage2/`; share that folder with its images together.
- The generated manifest records package versions and input hashes. Inputs and Stage 1
  outputs are unchanged. Acknowledge AI assistance in the final report appendix.
"""
    (out / "stage2_report.md").write_text(report)
    manifest = {
        "assets": list(ASSETS), "start": START, "end": END, "months": len(excess),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "hac_lags": HAC_LAGS, "data_sha256": {
            name: hashlib.sha256((config.DATA_DIR / name).read_bytes()).hexdigest()
            for name in ("prices_us_monthly.csv", "ff3_monthly.csv")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main():
    print(f"Stage 2 report: {run() / 'stage2_report.md'}")


if __name__ == "__main__":
    main()
