"""Visualisation layer.

One place for the report's look (seaborn with the QuantumBlack dark theme) and
one function per story beat.  The functions take plain data or the result
objects from :mod:`goldman.analysis` and return the path of the figure they
write, so the notebook only has to call them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; imported before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import qbstyles  # noqa: E402
import seaborn as sns  # noqa: E402

__all__ = [
    "COLORS",
    "apply_theme",
    "plot_correlation",
    "plot_risk_return",
    "plot_weights",
    "plot_weight_drift",
    "plot_robustness",
    "plot_alpha_intervals",
    "plot_sml",
    "plot_grs",
    "plot_sml_fit",
    "plot_proxy_comparison",
]

COLORS = {
    "long": "#4c9be8",
    "short": "#e4572e",
    "accent": "#f4a261",
    "theory": "#e9ecef",
    "positive": "#5fbf77",
    "muted": "#94a3b8",
    "grid": "#2b3d47",
}

MARKET_LABELS = {"us": "United States", "sg": "Singapore"}
VARIANT_LABELS = {
    "unconstrained": "unconstrained",
    "long_only": "no short",
    "shrunk": "shrunk",
    "shrunk_long_only": "shrunk, no short",
}
PROXY_LABELS = {"value_weighted": "own index", "ff_market": "Fama and French"}


def apply_theme(dark: bool = True, context: str = "notebook") -> None:
    """Apply the shared seaborn + QuantumBlack theme.

    Safe to call repeatedly.  ``minor_ticks`` is off because the QuantumBlack
    minor tick labels overlap the major ones at this figure size.
    """
    sns.set_theme(style="darkgrid", context=context, palette=list(COLORS.values()))
    qbstyles.mpl_style(dark=dark, minor_ticks=False)
    plt.rcParams.update(
        {
            "figure.autolayout": False,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "font.size": 9,
            "savefig.facecolor": plt.rcParams.get("savefig.facecolor", "#0C1C23"),
        }
    )


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _label(name: str, mapping: dict[str, str]) -> str:
    return mapping.get(name, name)


# --------------------------------------------------------------- data section
def plot_correlation(returns: pd.DataFrame, output_path: str | Path) -> Path:
    """Correlation heatmap of the universe."""
    apply_theme()
    correlation = returns.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

    fig, axis = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        correlation,
        mask=mask,
        cmap=cmap,
        center=0.0,
        square=True,
        linewidths=0.3,
        linecolor="#0C1C23",
        cbar_kws={"shrink": 0.7, "label": "correlation"},
        ax=axis,
    )
    axis.set_title("Monthly return correlations")
    axis.tick_params(labelsize=7)
    return _save(fig, output_path)


# --------------------------------------------------------- Stage 1 portfolio
def plot_risk_return(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Annualised mean against volatility, coloured by Sharpe."""
    apply_theme()
    fig, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        summary["vol (ann.)"],
        summary["mean (ann.)"],
        c=summary["sharpe (ann.)"],
        cmap="crest",
        s=90,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for asset, row in summary.iterrows():
        axis.annotate(
            asset,
            (row["vol (ann.)"], row["mean (ann.)"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
            color=COLORS["muted"],
        )
    fig.colorbar(scatter, ax=axis, shrink=0.8, label="Sharpe ratio")
    axis.set_xlabel("annualised volatility")
    axis.set_ylabel("annualised mean return")
    axis.set_title("Risk and return across the universe")
    return _save(fig, output_path)


def plot_weights(
    weights: pd.DataFrame,
    output_path: str | Path,
    comparison_label: str = "comparison",
    title: str = "Estimated tangency weights versus the index",
) -> Path:
    """Tangency weights above the comparison weights, sharing the asset order."""
    apply_theme()
    ordered = weights.reindex(weights["tangency"].abs().sort_values(ascending=False).index)
    positions = np.arange(len(ordered))

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    top.bar(positions, ordered["tangency"], 0.7, color=COLORS["long"], label="tangency")
    top.bar(
        positions,
        ordered["tangency"].where(ordered["tangency"] < 0, 0.0),
        0.7,
        color=COLORS["short"],
        label="short leg",
    )
    top.axhline(0, lw=0.8, color="white")
    top.set_ylabel("tangency weight")
    top.legend(loc="upper right")
    top.set_title(title)

    bottom.bar(
        positions, ordered["comparison"], 0.7, color=COLORS["accent"], label=comparison_label
    )
    bottom.axhline(0, lw=0.8, color="white")
    bottom.set_ylabel("index")
    bottom.legend(loc="upper right")
    bottom.set_xticks(positions)
    bottom.set_xticklabels(ordered.index, rotation=90, fontsize=8)
    return _save(fig, output_path)


def plot_weight_drift(
    drift: pd.DataFrame,
    output_path: str | Path,
    label: str = "",
    title: str = "Tangency weights across subsamples",
) -> Path:
    """First-half weights against second-half weights, with the identity line."""
    apply_theme()
    x = drift["first half"]
    y = drift["second half"]
    correlation = float(x.corr(y))

    fig, axis = plt.subplots(figsize=(7, 6.5))
    axis.scatter(x, y, s=55, color=COLORS["long"], edgecolor="white", linewidth=0.4, zorder=3)
    limits = [
        float(min(x.min(), y.min())) - 0.3,
        float(max(x.max(), y.max())) + 0.3,
    ]
    axis.plot(limits, limits, ls="--", lw=1.3, color=COLORS["theory"], label="identical weights")
    axis.axhline(0, lw=0.7, color=COLORS["muted"])
    axis.axvline(0, lw=0.7, color=COLORS["muted"])
    for asset in drift.index:
        axis.annotate(
            asset,
            (x[asset], y[asset]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
            color=COLORS["muted"],
        )
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("weight estimated on first half")
    axis.set_ylabel("weight estimated on second half")
    axis.set_title(f"{title}{f' ({label.upper()})' if label else ''}")
    axis.legend(loc="upper left")
    axis.text(
        0.98,
        0.02,
        f"correlation {correlation:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["accent"],
    )
    return _save(fig, output_path)


def plot_robustness(
    diagnostics: pd.DataFrame,
    output_path: str | Path,
    title: str = "Taming the tangency portfolio",
) -> Path:
    """How extreme the weights are, and how the variants score out of sample."""
    apply_theme()
    labels = [_label(name, VARIANT_LABELS) for name in diagnostics.index]
    positions = np.arange(len(labels))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    longest = diagnostics["max weight"].to_numpy()
    shortest = diagnostics["min weight"].to_numpy()
    shorts = diagnostics["shorts"].to_numpy(dtype=int)

    left.bar(positions - 0.2, longest, 0.4, color=COLORS["long"], label="largest long")
    left.bar(positions + 0.2, shortest, 0.4, color=COLORS["short"], label="largest short")
    left.axhline(0, lw=0.8, color="white")
    left.set_title("Largest positions")
    left.set_ylabel("portfolio weight")
    for x, count in zip(positions, shorts, strict=True):
        left.annotate(
            f"{count} shorts", (x, shortest.min() - 0.12), ha="center", fontsize=8,
            color=COLORS["muted"],
        )
    left.legend(loc="upper right")

    right.bar(
        positions, diagnostics["out of sample Sharpe"], 0.55, color=COLORS["positive"]
    )
    right.axhline(0, lw=0.8, color="white")
    right.set_title("Out of sample Sharpe")
    right.set_ylabel("annualised Sharpe")

    for axis in (left, right):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=15, ha="right")
    fig.suptitle(title)
    return _save(fig, output_path)


# -------------------------------------------------------------- Stage 2 CAPM
def plot_alpha_intervals(
    table: pd.DataFrame,
    output_path: str | Path,
    title: str = "Alphas with 95% confidence intervals",
) -> Path:
    """Alpha per asset with its confidence interval, coloured by significance."""
    apply_theme()
    ordered = table.sort_values("alpha")
    positions = np.arange(len(ordered))
    significant = ordered["p_alpha"] < 0.05
    colors = np.where(significant, COLORS["short"], COLORS["long"])

    fig, axis = plt.subplots(figsize=(8.5, 8))
    axis.errorbar(
        ordered["alpha"],
        positions,
        xerr=1.96 * ordered["se_alpha"],
        fmt="none",
        ecolor=COLORS["muted"],
        elinewidth=1.0,
        capsize=2,
        zorder=2,
    )
    axis.scatter(
        ordered["alpha"], positions, c=colors, s=46, zorder=3,
        edgecolor="white", linewidth=0.4,
    )
    axis.axvline(0, lw=1.1, color="white")
    axis.set_yticks(positions)
    axis.set_yticklabels(ordered.index, fontsize=7)
    axis.set_xlabel("monthly alpha")
    axis.set_title(title)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLORS["short"], label="significant at 5%"),
        plt.Line2D([], [], marker="o", ls="", color=COLORS["long"], label="not significant"),
    ]
    axis.legend(handles=handles, loc="lower right")
    return _save(fig, output_path)


def plot_sml(
    table: pd.DataFrame,
    fit,
    output_path: str | Path,
    title: str = "Empirical security market line",
    fitted_label: str = "fitted SML",
) -> Path:
    """Average excess return against beta, with fitted and theoretical lines."""
    apply_theme()
    beta = table["beta"].to_numpy()
    mean_excess = table["mean_excess"].to_numpy()
    grid = np.linspace(beta.min() - 0.1, beta.max() + 0.1, 50)

    fig, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        beta,
        mean_excess,
        c=table["r_squared"],
        cmap="crest",
        s=80,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for asset, x, y in zip(table.index, beta, mean_excess, strict=True):
        axis.annotate(
            asset, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points",
            color=COLORS["muted"],
        )
    axis.plot(
        grid,
        fit.intercept + fit.slope * grid,
        color=COLORS["accent"],
        lw=2,
        label=f"{fitted_label} (slope {fit.slope:.3f}, intercept {fit.intercept:.3f})",
    )
    axis.plot(
        grid,
        fit.theoretical_slope * grid,
        color=COLORS["theory"],
        ls="--",
        lw=1.6,
        label=f"CAPM theory (slope {fit.theoretical_slope:.3f})",
    )
    axis.axhline(0, color=COLORS["muted"], lw=0.7)
    fig.colorbar(scatter, ax=axis, shrink=0.8, label="R squared")
    axis.set_xlabel(r"estimated $\hat\beta_i$")
    axis.set_ylabel(r"average excess return $\bar{R}_i - \bar{R}_f$")
    axis.set_title(title)
    axis.legend(loc="best")
    return _save(fig, output_path)


def plot_grs(results: dict, output_path: str | Path) -> Path:
    """GRS p-value per universe against the 5% line."""
    apply_theme()
    names = list(results)
    p_values = [float(results[name].grs["p_value"]) for name in names]
    positions = np.arange(len(names))
    colors = [
        COLORS["short"] if p < 0.05 else COLORS["positive"] for p in p_values
    ]

    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.bar(positions, p_values, 0.55, color=colors)
    axis.axhline(0.05, ls="--", lw=1.2, color=COLORS["accent"], label="5% line")
    for x, value in zip(positions, p_values, strict=True):
        axis.annotate(f"p = {value:.3f}", (x, value), ha="center", va="bottom",
                      fontsize=9, color=COLORS["theory"])
    axis.set_xticks(positions)
    axis.set_xticklabels([_label(name, MARKET_LABELS) for name in names])
    axis.set_ylim(0, max(max(p_values) * 1.25, 0.1))
    axis.set_ylabel("GRS p value")
    axis.set_title("Joint test that all alphas are zero")
    axis.legend(loc="upper right")
    return _save(fig, output_path)


def plot_sml_fit(results: dict, output_path: str | Path) -> Path:
    """Fitted versus theoretical SML slope and intercept by universe."""
    apply_theme()
    names = list(results)
    fitted_slope = [float(results[n].fit.slope) for n in names]
    theory_slope = [float(results[n].fit.theoretical_slope) for n in names]
    intercepts = [float(results[n].fit.intercept) for n in names]
    positions = np.arange(len(names))

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    left.bar(positions - 0.2, fitted_slope, 0.4, color=COLORS["long"], label="fitted")
    left.bar(positions + 0.2, theory_slope, 0.4, color=COLORS["accent"], label="theory")
    left.set_title("SML slope")
    left.set_ylabel("slope")
    left.legend(loc="upper right")

    right.bar(positions, intercepts, 0.55, color=COLORS["positive"])
    right.axhline(0, lw=1.0, color="white")
    right.set_title("SML intercept (theory = 0)")
    right.set_ylabel("intercept")

    for axis in (left, right):
        axis.set_xticks(positions)
        axis.set_xticklabels([_label(n, MARKET_LABELS) for n in names])
    fig.suptitle("Fitted line versus CAPM theory")
    return _save(fig, output_path)


def plot_proxy_comparison(results: dict, output_path: str | Path) -> Path:
    """Betas under two market proxies, and the resulting GRS p-values."""
    apply_theme()
    methods = list(results)
    betas = pd.DataFrame({m: results[m].table["beta"] for m in methods})
    left_name, right_name = methods[0], methods[1]
    p_values = [float(results[m].grs["p_value"]) for m in methods]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
    left.scatter(
        betas[left_name],
        betas[right_name],
        s=60,
        color=COLORS["long"],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    limits = [
        float(min(betas.min().min(), 0)) - 0.1,
        float(betas.max().max()) + 0.1,
    ]
    left.plot(limits, limits, ls="--", lw=1.2, color=COLORS["theory"], label="same beta")
    left.set_xlim(limits)
    left.set_ylim(limits)
    left.set_xlabel(f"beta, {_label(left_name, PROXY_LABELS)}")
    left.set_ylabel(f"beta, {_label(right_name, PROXY_LABELS)}")
    left.set_title("Betas move with the proxy")
    left.legend(loc="upper left")

    positions = np.arange(len(methods))
    right.bar(positions, p_values, 0.5, color=COLORS["positive"])
    right.axhline(0.05, ls="--", lw=1.2, color=COLORS["accent"], label="5% line")
    for x, value in zip(positions, p_values, strict=True):
        right.annotate(
            f"p = {value:.3f}", (x, value), ha="center", va="bottom",
            fontsize=9, color=COLORS["theory"],
        )
    right.set_xticks(positions)
    right.set_xticklabels([_label(m, PROXY_LABELS) for m in methods])
    right.set_ylim(0, max(max(p_values) * 1.3, 0.1))
    right.set_ylabel("GRS p value")
    right.set_title("And so does the verdict")
    right.legend(loc="upper right")
    return _save(fig, output_path)
