"""US21 meeting pack. Offline, fixed asset selection, no shared data mutations."""

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
from ..data import load_ff3, load_prices, load_value_weights
from .portfolio import equal_weights, excess_means, subsample_tangency, tangency_weights
from .portfolio_choice import summary_statistics

# Selection by instrument type, specified before portfolio estimation.
ASSETS = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "BAC", "JNJ",
    "PG", "XOM", "CVX", "KO", "PEP", "WMT", "HD", "DIS", "CSCO", "INTC",
    "MRK", "VZ",
)
START, END = "2016-10", "2026-07"
TRAIN_MONTHS = 96
FACTOR_SOURCE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html"


def inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Require a complete fixed monthly window; never fill missing asset returns/RF."""
    prices = load_prices(universe="us").loc[:, list(ASSETS)]
    prices.index = prices.index.to_period("M")
    required_prices = pd.period_range(pd.Period(START) - 1, END, freq="M")
    if not prices.index.is_unique:
        raise ValueError("Duplicate price months")
    prices = prices.reindex(required_prices)
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Selected price panel must be complete and positive")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    factors = load_ff3().reindex(returns.index)
    if factors.isna().any().any() or len(returns) < 96:
        raise ValueError("Require observed factors/RF and at least 96 return months")
    # Cache rows are labelled by month-start but describe that month's closing prices.
    # Fetch the preceding snapshot explicitly so October uses September's weights.
    snapshots = load_value_weights(
        universe="us", assets=list(ASSETS), dates=required_prices.to_timestamp()
    )
    snapshots.index = snapshots.index.to_period("M")
    beginning_weights = lag_weights(snapshots, returns.index)
    return returns, factors, snapshots.loc[returns.index], beginning_weights


def lag_weights(snapshots: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Map each return month to the prior calendar month's weight snapshot."""
    lagged = snapshots.reindex(months - 1).copy()
    lagged.index = months
    if lagged.isna().any().any():
        raise ValueError("Missing previous-month value weights")
    return lagged.div(lagged.sum(axis=1), axis=0)


def estimate(returns: pd.DataFrame, rf: pd.Series) -> pd.Series:
    cov = returns.cov()
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Covariance must be positive definite")
    means = excess_means(returns, rf)
    raw = np.linalg.solve(cov.to_numpy(), means.to_numpy())
    if raw.sum() <= 1e-10:
        raise ValueError("No positive-normalization tangency solution in this sample")
    weights = tangency_weights(means, cov)
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid tangency weights")
    return weights


def portfolio_returns(
    returns: pd.DataFrame, weights: pd.Series | pd.DataFrame
) -> pd.Series:
    """Beginning-of-month target weights; fixed vectors imply monthly rebalancing."""
    aligned = weights.reindex(columns=returns.columns, index=returns.index) if isinstance(
        weights, pd.DataFrame
    ) else weights.reindex(returns.columns)
    if np.isnan(aligned.to_numpy()).any():
        raise ValueError("Missing aligned portfolio weights")
    return returns.mul(aligned, axis="columns").sum(axis=1, min_count=len(returns.columns))


def metrics(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Arithmetic annual mean; sample volatility; excess-return Sharpe; initial NAV=1."""
    excess = returns.sub(rf, axis=0)
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1)
    if (returns <= -1).any().any():
        raise ValueError("Portfolio loses at least 100%; compound-growth statistics invalid")
    return pd.DataFrame({
        "annual_mean": returns.mean() * 12,
        "annual_volatility": returns.std(ddof=1) * np.sqrt(12),
        "sharpe": excess.mean() / excess.std(ddof=1) * np.sqrt(12),
        "cagr": wealth.iloc[-1] ** (12 / len(returns)) - 1,
        "max_drawdown": (wealth / peak - 1).min(),
        "worst_month": returns.min(),
        "months": len(returns),
    })


def metric_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Portfolio | Annual mean | Annual volatility | Sharpe | CAGR | Max drawdown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in frame.iterrows():
        lines.append(
            f"| {label} | {row.annual_mean:.1%} | {row.annual_volatility:.1%} | "
            f"{row.sharpe:.2f} | {row.cagr:.1%} | {row.max_drawdown:.1%} |"
        )
    return "\n".join(lines)


def charts(weights: pd.DataFrame, drift: pd.DataFrame, holdout: pd.DataFrame, out: Path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    weights["tangency"].plot.bar(ax=axes[0], color="#27647b", rot=90)
    weights[["value_average", "equal"]].plot.bar(
        ax=axes[1], color=["#d89939", "#9ca7ad"], rot=90
    )
    axes[0].set_title("US21 | Fitted tangency weights versus simple benchmarks", loc="left")
    axes[0].set_ylabel("Tangency weight")
    axes[1].set_ylabel("Benchmark weight")
    axes[1].legend(["Value weighted: average composition", "Equal weighted"], ncol=2)
    for ax in axes:
        ax.axhline(0, color="#444444", lw=0.7)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("")
    fig.savefig(out / "weights.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    drift[["first half", "second half"]].plot.bar(ax=ax, color=["#27647b", "#d89939"])
    ax.axhline(0, color="#444444", lw=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("US21 | Does changing the estimation window change the portfolio?", loc="left")
    ax.set_ylabel("Tangency weight")
    ax.set_xlabel("")
    fig.savefig(out / "stability.png", dpi=150)
    plt.close(fig)

    wealth = (1 + holdout).cumprod()
    wealth.index = wealth.index.to_timestamp(how="end")
    initial = pd.DataFrame(
        1., index=[wealth.index[0] - pd.offsets.MonthEnd(1)], columns=wealth.columns
    )
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    pd.concat([initial, wealth]).plot(ax=ax, lw=2)
    ax.set_title("US21 | Later-period check: weights estimated before the test period", loc="left")
    ax.set_ylabel("Growth of $1, before costs")
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    fig.savefig(out / "holdout.png", dpi=150)
    plt.close(fig)


def run(output_dir: Path | None = None) -> Path:
    out = Path(output_dir) if output_dir else config.OUTPUTS_DIR / "kai_us21"
    out.mkdir(parents=True, exist_ok=True)
    returns, factors, snapshots, lagged = inputs()
    rf = factors["RF"]
    tangent = estimate(returns, rf)
    equal = equal_weights(list(ASSETS))
    weights = pd.DataFrame({
        "tangency": tangent, "value_average": snapshots.mean(), "equal": equal,
    })
    drift = subsample_tangency(returns, rf)
    corr = float(drift["first half"].corr(drift["second half"]))
    portfolios = pd.DataFrame({
        "Tangency (fitted in sample)": portfolio_returns(returns, tangent),
        "Value weighted (lagged)": portfolio_returns(returns, lagged),
        "Equal weighted": portfolio_returns(returns, equal),
        "FF US market": factors["Mkt-RF"] + rf,
    })
    full_metrics = metrics(portfolios, rf)
    train, test = returns.iloc[:TRAIN_MONTHS], returns.iloc[TRAIN_MONTHS:]
    train_weights = estimate(train, rf.loc[train.index])
    holdout = portfolios.loc[test.index].drop(columns="Tangency (fitted in sample)")
    holdout.insert(0, "Tangency (training weights)", portfolio_returns(test, train_weights))
    holdout_metrics = metrics(holdout, rf.loc[test.index])
    asset_stats = summary_statistics(returns, rf)
    asset_excess = returns.sub(rf, axis=0)
    asset_stats["sharpe (ann.)"] = (
        asset_excess.mean() / asset_excess.std(ddof=1) * np.sqrt(12)
    ).round(3)
    frames = {
        "asset_returns": returns, "factors_aligned": factors,
        "summary_statistics": asset_stats, "weights": weights,
        "value_weights_end_month": snapshots, "value_weights_beginning_month": lagged,
        "subsample_weights": drift, "portfolio_returns_in_sample": portfolios,
        "comparison_in_sample": full_metrics, "holdout_weights": train_weights.to_frame(),
        "portfolio_returns_holdout": holdout, "comparison_holdout": holdout_metrics,
    }
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", float_format="%.12g")
    charts(weights, drift, holdout, out)
    largest = tangent.abs().idxmax()
    long = float(tangent.clip(lower=0).sum())
    short = float(-tangent.clip(upper=0).sum())
    extreme = returns.stack().abs().idxmax()
    extreme_return = returns.loc[extreme[0], extreme[1]]
    top = tangent.reindex(tangent.abs().sort_values(ascending=False).index).head(6)
    top_text = "\n".join(f"| {ticker} | {weight:+.1%} |" for ticker, weight in top.items())
    holdout_value_sharpe = holdout_metrics.loc['Value weighted (lagged)', 'sharpe']
    report = f"""# Kai's US21 portfolio: group meeting pack

## Selection and research question

Use all **21 individual stocks** in the supplied 30-asset US menu; exclude all nine sector ETFs.
This is an instrument-type rule, not a ranking of past winners. It avoids combining stocks with
funds that may hold the same stocks and makes the value-weight comparison easier to explain.
It remains a concentrated selection of large US companies, with survivorship/selection bias;
it is not the whole US market and is not evenly diversified across sectors.

**Assets:** {', '.join(ASSETS)}.

**Question:** How different is the estimated tangency portfolio from value and equal weighting,
and does its apparent advantage survive changing the estimation period?

## Main findings to discuss

- {len(ASSETS)} assets and {len(returns)} complete monthly returns, {START} through {END}.
- Largest fitted position: **{largest} {tangent[largest]:+.1%}**.
- Long exposure **{long:.1%}**, short exposure **{short:.1%}**,
  gross exposure **{long + short:.1%}**;
  net risky-asset weights sum to 100%. Negative weights mean short positions.
- Half-sample weight correlation: **{corr:.2f}**. The windows are
  {returns.index[0]}–{returns.index[len(returns)//2-1]} and
  {returns.index[len(returns)//2]}–{returns.index[-1]}.
- Full-sample tangency Sharpe is **{full_metrics.iloc[0].sharpe:.2f}**, measured on the same data
  used to estimate it. The later-period check gives **{holdout_metrics.iloc[0].sharpe:.2f}**.
  These are different periods; neither establishes future superiority.

| Largest absolute fitted positions | Weight |
|---|---:|
{top_text}

![Weight comparison](weights.png)

**Takeaway:** The fitted tangency portfolio uses both long and short positions. The benchmark
panel has a separate scale; compare signs and concentration as well as raw bar heights.

## Full-sample comparison: descriptive, fitted with hindsight

{metric_table(full_metrics)}

Tangency weights are estimated using all {len(returns)} months and rebalanced to those weights
each month in this descriptive calculation. Equal weights are also rebalanced monthly.
Value-weighted returns use the previous month's market-cap weights, renormalized over these
21 assets. The chart shows average **end-of-month** value weights for composition only;
those averages are not used to calculate the value-weighted return series.

## Stability and later-period check

![Subsample stability](stability.png)

**Takeaway:** The weight correlation is {corr:.2f}; changing the estimation window changes the
portfolio. Estimation error and changing market conditions can both contribute.

For an extra check, estimate tangency weights once on the first **{TRAIN_MONTHS} months**
({train.index[0]}–{train.index[-1]}), then rebalance to that fixed vector during the remaining
**{len(test)} months** ({test.index[0]}–{test.index[-1]}). No test-period returns enter that fit.
This is a fixed chronological holdout, not a rolling strategy or a fully unbiased backtest.

{metric_table(holdout_metrics)}

![Later-period comparison](holdout.png)

**Takeaway:** On the {len(test)}-month check, the fitted-training portfolio has Sharpe
{holdout_metrics.iloc[0].sharpe:.2f}, versus {holdout_value_sharpe:.2f}
for value weighting and {holdout_metrics.loc['Equal weighted', 'sharpe']:.2f} for equal weighting.
The short period, preselected surviving stocks and approximate historical caps limit inference.

## Conventions and material caveats

- USD; simple monthly returns from the group's cached adjusted closes; no new downloads.
- RF is the observed monthly US Treasury-bill proxy in the supplied FF3 file. August 2026
  is excluded because factor/RF data stop in July; no constant fallback is used.
- The market reference is `Mkt-RF + RF`. Later CAPM regressions should use `Mkt-RF` directly.
  [Official factor definitions]({FACTOR_SOURCE}).
- Annual mean = 12 × monthly arithmetic mean. Annual volatility = sqrt(12) × sample standard
  deviation of total returns. Sharpe = sqrt(12) × mean excess return / sample standard deviation
  of excess returns. CAGR compounds returns; it is different from annual arithmetic mean.
- Drawdown includes the initial wealth of 1. All results exclude fees, bid–ask costs,
  stock-borrow costs, taxes, margin constraints and financing frictions.
- The cached market caps carry early share counts backward where history is absent, notably
  META before July 2022. Lagging weights prevents same-month price leakage but cannot remove
  that underlying approximation. The shared cache has not been independently reconstructed.
- Largest absolute asset-month return in this selection is **{extreme[1]} {extreme_return:+.1%}
  in {extreme[0]}**. It is retained unchanged from the supplied data; verify the source/corporate
  action before interpreting any result driven by this observation. No winsorization was applied.
- The selected universe is fixed before fitting this analysis, but the supplied universe itself
  uses surviving names; the holdout does not remove that bias.
- This is Stage 1 research for comparison. CAPM, factor regressions and the final investment
  committee recommendation are still to be completed by the group.

## Bring to the group meeting

1. Explain the selection rule: all 21 individual stocks, no sector ETFs.
2. Compare on the same dates, RF, rebalancing rule, short-selling assumptions and metrics.
3. Show `weights.csv`, this report, and the stability plot. Discuss gross exposure, not just Sharpe.
4. Decide whether to keep this universe, add a long-only robustness check, or combine ideas.
5. Agree a market proxy for Stage 2 and an alternative proxy for Stage 4.

Reproduce with `python run_kai_us21.py` from the checkout using its project dependencies;
or run `notebooks/kai_us21.ipynb` top to bottom. Results go only to `outputs/kai_us21/`.
Raw data and the existing shared notebook are unchanged. Outputs are git-ignored; rerun or
share this folder separately. Record AI assistance in the final report appendix, e.g.
"Codex assisted with the US21 analysis code, validation, and drafting; the group reviewed results."
"""
    (out / "meeting_report.md").write_text(report)
    manifest = {
        "selection": list(ASSETS), "start": START, "end": END,
        "months": len(returns), "train_months": TRAIN_MONTHS,
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "data_sha256": {
            name: hashlib.sha256((config.DATA_DIR / name).read_bytes()).hexdigest()
            for name in ("prices_us_monthly.csv", "ff3_monthly.csv", "value_weights_us.csv")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main():
    print(f"Meeting pack: {run() / 'meeting_report.md'}")


if __name__ == "__main__":
    main()
