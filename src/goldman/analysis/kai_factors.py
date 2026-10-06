"""Stage 3: size/value factors and a matched-sample comparison with CAPM."""

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
from .kai_capm import HAC_LAGS, fit_capm, hac_covariance, load_inputs
from .kai_us21 import ASSETS, END, FACTOR_SOURCE, START

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML"]


def fit_factor_model(
    excess: pd.DataFrame, factors: pd.DataFrame, lags: int = HAC_LAGS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OLS with an intercept; inputs are asset excess returns and factor returns."""
    if not excess.index.equals(factors.index) or not excess.index.is_unique:
        raise ValueError("Asset and factor dates must align uniquely")
    if not excess.columns.is_unique or not factors.columns.is_unique or factors.shape[1] == 0:
        raise ValueError("Need unique assets and at least one uniquely named factor")
    y = excess.to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(factors)), factors.to_numpy(dtype=float)])
    n, k = x.shape
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Regression inputs must be finite")
    if n <= k or np.linalg.matrix_rank(x) < k:
        raise ValueError("Insufficient observations or collinear factors")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    sse = (residuals**2).sum(axis=0)
    total = ((y - y.mean(axis=0))**2).sum(axis=0)
    if (total <= 0).any():
        raise ValueError("Asset excess returns must vary")
    se = np.sqrt(np.diag(np.linalg.inv(x.T @ x))[:, None] * sse / (n - k))
    robust = np.column_stack([
        np.sqrt(np.maximum(np.diag(hac_covariance(x, residuals[:, i], lags)), 0))
        for i in range(y.shape[1])
    ])
    table = pd.DataFrame({
        "alpha_monthly": coefficients[0], "alpha_annual": coefficients[0] * 12,
        "alpha_se_monthly": se[0], "alpha_se_hac_monthly": robust[0],
        "t_alpha": coefficients[0] / se[0], "t_alpha_hac": coefficients[0] / robust[0],
        "r_squared": 1 - sse / total,
        "adjusted_r_squared": 1 - (sse / (n - k)) / (total / (n - 1)),
        "n_months": n, "residual_df": n - k,
    }, index=excess.columns)
    for i, name in enumerate(factors.columns, start=1):
        table[f"loading_{name}"] = coefficients[i]
        table[f"t_{name}"] = coefficients[i] / se[i]
        table[f"hac_t_{name}"] = coefficients[i] / robust[i]
    table["flag_abs_t_gt_2"] = table.t_alpha.abs() > 2
    table["flag_abs_hac_t_gt_2"] = table.t_alpha_hac.abs() > 2
    table.index.name = "asset"
    return table, pd.DataFrame(residuals, index=excess.index, columns=excess.columns)


def anomaly_status(before_alpha: float, after_alpha: float, before_t: float, after_t: float) -> str:
    """Separate a smaller alpha from simply crossing a statistical threshold."""
    before, after = abs(before_t) > 2, abs(after_t) > 2
    if before and after:
        return "survives"
    if before:
        return "attenuated; no longer flagged" if abs(after_alpha) < abs(before_alpha) else (
            "no longer flagged; magnitude not reduced"
        )
    return "newly flagged" if after else "not flagged in either model"


def compare(capm: pd.DataFrame, ff3: pd.DataFrame) -> pd.DataFrame:
    if not capm.index.equals(ff3.index) or not capm.n_months.equals(ff3.n_months):
        raise ValueError("Models must use the same assets and sample size")
    result = pd.DataFrame(index=capm.index)
    for prefix, frame in [("capm", capm), ("ff3", ff3)]:
        for name in ["alpha_annual", "t_alpha", "t_alpha_hac", "r_squared"]:
            result[f"{prefix}_{name}"] = frame[name]
    result["change_abs_alpha_annual"] = ff3.alpha_annual.abs() - capm.alpha_annual.abs()
    result["capm_adjusted_r_squared"] = (
        1 - (1 - capm.r_squared) * (capm.n_months - 1) / (capm.n_months - 2)
    )
    result["ff3_adjusted_r_squared"] = ff3.adjusted_r_squared
    for name, t in [("classical_verdict", "t_alpha"), ("hac_verdict", "t_alpha_hac")]:
        result[name] = [anomaly_status(capm.loc[a, "alpha_annual"], ff3.loc[a, "alpha_annual"],
                                       capm.loc[a, t], ff3.loc[a, t]) for a in capm.index]
    return result


def draw_charts(comparison: pd.DataFrame, out: Path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ordered = comparison.sort_values("capm_alpha_annual")
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    ax.barh(positions - .18, ordered.capm_alpha_annual, .35, label="CAPM", color="#27647b")
    ax.barh(positions + .18, ordered.ff3_alpha_annual, .35,
            label="Market + size + value", color="#d89939")
    ax.set_yticks(positions, ordered.index)
    ax.axvline(0, color="#666666", lw=.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Annualized alpha (monthly intercept × 12)",
           title="US21 | Do size and value reduce unexplained average returns?")
    ax.legend()
    fig.savefig(out / "alpha_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    display = comparison[["capm_adjusted_r_squared", "ff3_adjusted_r_squared"]].copy()
    display.columns = ["CAPM", "Market + size + value"]
    display.plot.bar(ax=ax, color=["#27647b", "#d89939"])
    ax.set(xlabel="", ylabel="Adjusted R-squared",
           title="US21 | Does model fit improve after allowing for more factors?")
    fig.savefig(out / "adjusted_r_squared.png", dpi=150)
    plt.close(fig)


def comparison_markdown(comparison: pd.DataFrame) -> str:
    lines = ["| Asset | CAPM alpha/yr | FF3 alpha/yr | CAPM t | FF3 t | FF3 HAC t |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, r in comparison.iterrows():
        lines.append(f"| {name} | {r.capm_alpha_annual:.2%} | {r.ff3_alpha_annual:.2%} | "
                     f"{r.capm_t_alpha:.2f} | {r.ff3_t_alpha:.2f} | {r.ff3_t_alpha_hac:.2f} |")
    return "\n".join(lines)


def flagged_names(table: pd.DataFrame, column: str) -> str:
    return ", ".join(table.index[table[column].abs() > 2]) or "none"


def run(output_dir: Path | None = None) -> Path:
    out = Path(output_dir) if output_dir else config.OUTPUTS_DIR / "kai_us21_stage3"
    out.mkdir(parents=True, exist_ok=True)
    returns, factors = load_inputs()
    excess = returns.sub(factors.RF, axis=0)
    capm, _ = fit_capm(excess, factors["Mkt-RF"])
    ff3, residuals = fit_factor_model(excess, factors[FACTOR_COLUMNS])
    comparison = compare(capm, ff3)
    hac_sensitivity = pd.DataFrame({
        f"lags_{lag}": fit_factor_model(excess, factors[FACTOR_COLUMNS], lags=lag)[0].t_alpha_hac
        for lag in (0, HAC_LAGS, 6)
    })
    n = len(excess)
    # Components sum to the observed mean return, providing a check on alpha interpretation.
    decomposition = pd.DataFrame({"alpha_annual": ff3.alpha_annual})
    for name in FACTOR_COLUMNS:
        decomposition[f"{name}_contribution_annual"] = (
            ff3[f"loading_{name}"] * factors[name].mean() * 12
        )
    decomposition["observed_mean_excess_annual"] = excess.mean() * 12
    for name, frame in {
        "capm_results_matched": capm, "ff3_results": ff3, "alpha_comparison": comparison,
        "ff3_residuals": residuals, "mean_return_decomposition": decomposition,
        "factors_aligned": factors, "asset_excess_returns": excess,
        "factor_correlations": factors[FACTOR_COLUMNS].corr(),
        "hac_lag_sensitivity": hac_sensitivity,
    }.items():
        frame.to_csv(out / f"{name}.csv", float_format="%.12g")
    draw_charts(comparison, out)
    shrunk = int((comparison.change_abs_alpha_annual < 0).sum())
    improved = int((comparison.ff3_adjusted_r_squared > comparison.capm_adjusted_r_squared).sum())
    mean_before, mean_after = capm.alpha_annual.abs().mean(), ff3.alpha_annual.abs().mean()
    verdict_rows = ["| Stage 2 flagged asset | Classical result | HAC result |",
                    "|---|---|---|"]
    original_flags = capm.index[(capm.t_alpha.abs() > 2) | (capm.t_alpha_hac.abs() > 2)]
    for name in original_flags:
        verdict_rows.append(f"| {name} | {comparison.loc[name, 'classical_verdict']} | "
                            f"{comparison.loc[name, 'hac_verdict']} |")
    if not len(original_flags):
        verdict_rows.append("| None | No Stage 2 flag | No Stage 2 flag |")
    survived = comparison.index[comparison.classical_verdict == "survives"].tolist()
    attenuated = comparison.index[
        comparison.classical_verdict == "attenuated; no longer flagged"
    ].tolist()
    new = comparison.index[comparison.classical_verdict == "newly flagged"].tolist()
    loading_rows = "\n".join(
        f"| {a} | {ff3.loc[a, 'loading_Mkt-RF']:.3f} | "
        f"{ff3.loc[a, 'loading_SMB']:.3f} | {ff3.loc[a, 'loading_HML']:.3f} |"
        for a in original_flags
    )
    summary = {
        "months": n, "assets": len(ASSETS), "additional_factors": ["SMB", "HML"],
        "classical_surviving_stage2_flags": survived,
        "classical_attenuated_stage2_flags": attenuated, "classical_new_flags": new,
        "assets_with_smaller_absolute_alpha": shrunk,
        "mean_absolute_annual_alpha_capm": float(mean_before),
        "mean_absolute_annual_alpha_ff3": float(mean_after),
        "assets_with_improved_adjusted_r_squared": improved,
        "formal_joint_alpha_test": False,
    }
    (out / "verdict.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# Kai US21 - Stage 3: market, size and value factors

## Findings for the group meeting

We add **SMB (size)** and **HML (value)** to the market model for the same **21 stocks**
and **{n} months, {START}–{END}**. Both models are estimated again on exactly the same
observations, so changes in alpha reflect the model change rather than a changed sample.

- Absolute alpha falls for **{shrunk} of 21** stocks.
- Mean absolute annualized alpha changes from **{mean_before:.2%}** to **{mean_after:.2%}**.
- Adjusted R² improves for **{improved} of 21** stocks.
- Classical Stage 2 anomalies that remain flagged: **{', '.join(survived) or 'none'}**.
- Classical Stage 2 anomalies with smaller alphas and no remaining flag:
  **{', '.join(attenuated) or 'none'}**.
- Newly flagged classical alphas: **{', '.join(new) or 'none'}**.

## Exhibit 3a - What the additional factors represent

**Size (SMB).** The published US factor uses six value-weighted portfolios formed from
two size groups and three book-to-market groups. SMB takes the average of the three small-firm
portfolio returns and subtracts the average of the three large-firm returns. A positive
loading indicates exposure to small-company performance relative to large companies.

**Value (HML).** Using the same six portfolios, HML averages the two high-book-to-market
portfolio returns and subtracts the average of the two low-book-to-market returns.
It measures value relative to growth. A negative loading indicates growth-like exposure;
the sign of its contribution also depends on the factor's average return over this sample.

These are existing factors from the supplied Ken French data, not factors constructed by
our group. We use their monthly decimal returns directly. SMB and HML are long–short
spreads: **do not subtract RF from them**. [Source: official factor definitions]({FACTOR_SOURCE}).

The fitted equation is:

`Ri − Rf = alpha + beta × (Rm − Rf) + s × SMB + h × HML + residual`.

We re-examine individual assets, as permitted by the handout. Sorted portfolios are an
optional extension; we have not sorted on full-sample outcomes or created new factor data.

## Exhibit 3b - Alphas before and after adding factors

{comparison_markdown(comparison)}

Annual alpha is the monthly intercept multiplied by 12. t-statistics are calculated from
monthly estimates and are not annualized. Classical errors use SSE/(T−4) for FF3 versus
SSE/(T−2) for CAPM. The HAC column uses the same Newey-West convention as Stage 2:
{HAC_LAGS} monthly lags, Bartlett weights and T/(T−k) correction including the intercept.

![Alpha comparison](alpha_comparison.png)

**Takeaway:** Adding size and value reduces absolute alpha for {shrunk}/21 stocks; the mean
absolute annualized alpha changes from {mean_before:.2%} to {mean_after:.2%}. More factors do
not guarantee smaller intercepts. Examine alpha magnitude and uncertainty together.

## Exhibit 3c - Which Stage 2 anomalies are absorbed or survive?

{chr(10).join(verdict_rows)}

Classical CAPM flags: **{flagged_names(capm, 't_alpha')}**.
Classical FF3 flags: **{flagged_names(ff3, 't_alpha')}**.
HAC CAPM flags: **{flagged_names(capm, 't_alpha_hac')}**.
HAC FF3 flags: **{flagged_names(ff3, 't_alpha_hac')}**.

We use the same |t(alpha)| > 2 screening rule throughout. "Survives" means an asset
is flagged in both models under the stated error convention. "Attenuated; no longer
flagged" requires both a reduction in absolute alpha and a loss of the flag. It is
consistent with partial absorption; it is not proof that the true alpha is zero.
Losing significance without a smaller alpha is identified separately. A new flag can
arise because alpha changes or its standard error falls; it is not automatically a new
economic anomaly or a profitable trading opportunity.

**Nvidia:** Annualized alpha changes from {capm.loc['NVDA', 'alpha_annual']:.2%} to
{ff3.loc['NVDA', 'alpha_annual']:.2%}; classical t is {ff3.loc['NVDA', 't_alpha']:.2f} and HAC t
is {ff3.loc['NVDA', 't_alpha_hac']:.2f}. Size and value explain little of this alpha.
**Disney:** FF3 HAC t is {ff3.loc['DIS', 't_alpha_hac']:.4f}, close to the screening boundary.
Across 0, 3 and 6 HAC lags it ranges from {hac_sensitivity.loc['DIS'].min():.4f} to
{hac_sensitivity.loc['DIS'].max():.4f}. Treat a near-threshold flag as marginal evidence;
the unrounded estimates determine the flags, even when a displayed t rounds to -2.00.
`hac_lag_sensitivity.csv` reports the same check for every asset; 3 lags remain the baseline.

| Asset | Market loading | Size loading | Value loading |
|---|---:|---:|---:|
{loading_rows}

The complete loading and loading t-statistic table is in `ff3_results.csv`.
`mean_return_decomposition.csv` separates the estimated annual mean excess return into
alpha and loading × mean factor return contributions. Market beta is re-estimated when
SMB/HML are added, so the change in alpha cannot be attributed to SMB/HML alone while
holding beta fixed.

## Model fit and limits

![Adjusted R-squared](adjusted_r_squared.png)

**Takeaway:** Adjusted R² improves for {improved}/21 stocks. Ordinary R² cannot decrease
when regressors are added to the same OLS sample; adjusted R² accounts for the extra
parameters. Better fit of monthly variation does not imply that all average-return alphas
disappear, that the added factors are independently priced, or that CAPM has been validated.

These are individual screens, not multiplicity-adjusted tests or a joint zero-alpha test.
Around one false flag is expected among 21 correctly calibrated 5% individual tests under
the complete null. Correlated stocks, estimated parameters and the small selected universe
limit that heuristic. Neither model is being evaluated out of sample here.

The original universe has survivorship/selection bias. The inherited Intel April 2026
return of **{returns.loc[pd.Period('2026-04', freq='M'), 'INTC']:+.1%}** remains unverified and
unchanged. Conclusions are preliminary pending input verification and group review.
The cached July-2026 factor cutoff is retained for consistency with Stages 1 and 2.

## Reproduce and next step

Run `python run_kai_stage3.py` or `notebooks/kai_stage3.ipynb` top to bottom.
Outputs go to `outputs/kai_us21_stage3/`; share the complete folder so the images work.
Input data and earlier-stage outputs are unchanged. The manifest records input hashes,
factor definitions link, runtime versions and model conventions.

**Stage 4 remains:** compare an alternative market proxy, write the investment-committee
recommendation, and explain the least trustworthy finding. These Stage 3 results provide
evidence for that judgment; they do not settle it. Acknowledge AI coding, validation and
drafting assistance in the final appendix and review all findings as a group.
"""
    (out / "stage3_report.md").write_text(report)
    manifest = {
        "assets": list(ASSETS), "start": START, "end": END, "months": n,
        "factors": FACTOR_COLUMNS, "factor_source": FACTOR_SOURCE,
        "hac_lags": HAC_LAGS, "hac_small_sample_correction": "T/(T-k)",
        "alpha_annualization": "monthly alpha times 12", "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "data_sha256": {
            name: hashlib.sha256((config.DATA_DIR / name).read_bytes()).hexdigest()
            for name in ("prices_us_monthly.csv", "ff3_monthly.csv")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main():
    print(f"Stage 3 report: {run() / 'stage3_report.md'}")


if __name__ == "__main__":
    main()
