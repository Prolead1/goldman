"""Visualisation layer.

One place for the report's look (seaborn with the QuantumBlack dark theme) and
one function per story beat.  The functions take plain data or the result
objects from :mod:`goldman.analysis` and return the path of the figure they
write, so the notebook only has to call them.

The chart type is chosen to fit the question: a heatmap for the correlation
structure, scatters for relationships, a strip plot for the weight clouds, a
forest plot for the alphas, a step histogram for the joint alpha test, and a
slope chart for the fitted-versus-theory comparison.
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
    "plot_alpha_comparison",
    "plot_factor_cumulative",
    "plot_oos_growth",
    "plot_oos_sharpe",
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
UNIVERSE_COLORS = {"us": COLORS["long"], "sg": COLORS["positive"]}
OOS_LABELS = {
    "index": "value-weighted index",
    "equal_weight": "1/N",
    "unconstrained": "tangency (unconstrained)",
    "long_only": "tangency (no short)",
    "shrunk": "shrunk",
    "shrunk_long_only": "shrunk, no short",
}
#: One distinct colour per rule; the index is the bar to beat, the raw tangency
#: the cautionary case.
OOS_COLORS = {
    "index": COLORS["accent"],
    "equal_weight": COLORS["positive"],
    "unconstrained": COLORS["short"],
    "long_only": COLORS["long"],
    "shrunk": COLORS["muted"],
    "shrunk_long_only": "#9b7ede",
}
#: Benchmarks are drawn heavier so the estimated strategies read as challengers.
OOS_EMPHASIS = {"index": OOS_COLORS["index"], "equal_weight": OOS_COLORS["equal_weight"]}


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


def _color(name: str, index: int) -> str:
    return UNIVERSE_COLORS.get(name, list(COLORS.values())[index % len(COLORS)])


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
    """Scatter of index weight against tangency weight, one point per asset."""
    apply_theme()
    index_weight = weights["comparison"]
    tangency = weights["tangency"]
    colors = np.where(tangency >= 0, COLORS["long"], COLORS["short"])

    fig, axis = plt.subplots(figsize=(8.5, 7))
    axis.axhline(0, lw=0.8, color=COLORS["muted"])
    axis.axvline(0, lw=0.8, color=COLORS["muted"])
    axis.scatter(
        index_weight, tangency, c=colors, s=80, zorder=3,
        edgecolor="white", linewidth=0.5,
    )
    for asset in weights.index:
        axis.annotate(
            asset,
            (index_weight[asset], tangency[asset]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
            color=COLORS["muted"],
        )
    axis.set_xlabel(comparison_label)
    axis.set_ylabel("tangency weight")
    axis.set_title(title)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLORS["long"], label="long"),
        plt.Line2D([], [], marker="o", ls="", color=COLORS["short"], label="short"),
    ]
    axis.legend(handles=handles, loc="upper right")
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
    weights: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_path: str | Path,
    title: str = "Taming the tangency portfolio",
) -> Path:
    """Every asset weight under each construction, as a jittered strip plot."""
    apply_theme()
    generator = np.random.default_rng(0)
    variants = list(weights.columns)
    positions = np.arange(len(variants))

    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    for position, variant in zip(positions, variants, strict=True):
        values = weights[variant].to_numpy(dtype=float)
        jitter = generator.uniform(-0.18, 0.18, size=len(values))
        colors = np.where(values >= 0, COLORS["long"], COLORS["short"])
        axis.scatter(values, position + jitter, c=colors, s=22, alpha=0.8, zorder=3)
    axis.axvline(0, lw=1.0, color="white")

    labels = []
    for name in variants:
        largest = max(
            abs(diagnostics.loc[name, "max weight"]),
            abs(diagnostics.loc[name, "min weight"]),
        )
        sharpe = diagnostics.loc[name, "out of sample Sharpe"]
        labels.append(
            f"{_label(name, VARIANT_LABELS)}\n"
            f"max |weight| {largest:.0%}, Sharpe {sharpe:.2f}"
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("portfolio weight")
    axis.set_title(title)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLORS["long"], label="long"),
        plt.Line2D([], [], marker="o", ls="", color=COLORS["short"], label="short"),
    ]
    axis.legend(handles=handles, loc="upper right")
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
    """Step histogram of individual alpha t statistics, with the GRS verdict."""
    apply_theme()
    names = list(results)
    fig, axis = plt.subplots(figsize=(8.5, 4.6))
    for index, name in enumerate(names):
        t_stats = results[name].table["t_alpha"].to_numpy()
        p_value = float(results[name].grs["p_value"])
        axis.hist(
            t_stats,
            bins=12,
            histtype="step",
            lw=2.2,
            color=_color(name, index),
            label=f"{_label(name, MARKET_LABELS)} (GRS p = {p_value:.3f})",
        )
    axis.axvspan(-1.96, 1.96, color=COLORS["muted"], alpha=0.06)
    for bound in (-1.96, 1.96):
        axis.axvline(bound, ls=":", lw=1.2, color=COLORS["accent"])
    axis.axvline(0, lw=1.0, color="white")
    axis.set_xlabel(r"$t(\hat\alpha_i)$")
    axis.set_ylabel("number of assets")
    axis.set_title("Individual alpha t statistics")
    axis.legend(loc="upper right")
    return _save(fig, output_path)


def plot_sml_fit(results: dict, output_path: str | Path) -> Path:
    """Slope chart: how far each universe moves from theory to the fit."""
    apply_theme()
    names = list(results)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for index, name in enumerate(names):
        fit = results[name].fit
        color = _color(name, index)
        label = _label(name, MARKET_LABELS)
        left.plot(
            [0, 1], [fit.theoretical_slope, fit.slope], marker="o", lw=2.2, color=color, label=label
        )
        right.plot([0, 1], [0.0, fit.intercept], marker="o", lw=2.2, color=color, label=label)
    for axis, ylabel, heading in (
        (left, "slope", "SML slope"),
        (right, "intercept", "SML intercept"),
    ):
        axis.axhline(0, lw=0.7, color=COLORS["muted"])
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["theory", "fitted"])
        axis.set_xlim(-0.2, 1.2)
        axis.set_ylabel(ylabel)
        axis.set_title(heading)
    left.legend(loc="best")
    fig.suptitle("Fitted line versus CAPM theory")
    return _save(fig, output_path)


def plot_proxy_comparison(results: dict, output_path: str | Path) -> Path:
    """Beta under one market proxy against beta under the other."""
    apply_theme()
    methods = list(results)
    first, second = methods[0], methods[1]
    x = results[first].table["beta"]
    y = results[second].table["beta"]

    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    axis.scatter(x, y, s=70, color=COLORS["long"], edgecolor="white", linewidth=0.5, zorder=3)
    limits = [float(min(x.min(), y.min(), 0)) - 0.1, float(max(x.max(), y.max())) + 0.1]
    axis.plot(limits, limits, ls="--", lw=1.3, color=COLORS["theory"], label="same beta")
    for asset in (y - x).abs().sort_values(ascending=False).head(6).index:
        axis.annotate(
            asset, (x[asset], y[asset]), fontsize=7, xytext=(4, 3),
            textcoords="offset points", color=COLORS["muted"],
        )
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel(f"beta, {_label(first, PROXY_LABELS)}")
    axis.set_ylabel(f"beta, {_label(second, PROXY_LABELS)}")
    axis.set_title("Betas under two market proxies")
    verdict = "\n".join(
        f"{_label(m, PROXY_LABELS)}: GRS p = {float(results[m].grs['p_value']):.3f}"
        for m in methods
    )
    axis.text(
        0.03,
        0.97,
        verdict,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=COLORS["accent"],
        bbox={"facecolor": "#0C1C23", "edgecolor": COLORS["muted"], "alpha": 0.7},
    )
    axis.legend(loc="lower right")
    return _save(fig, output_path)


# ------------------------------------------------------------- Stage 3 factors
def plot_alpha_comparison(
    alphas: pd.DataFrame,
    summary: pd.DataFrame | None,
    output_path: str | Path,
    title: str = "Alphas before and after factors",
) -> Path:
    """Dumbbell chart: each asset's alpha under every factor model.

    Assets are ordered by the first model's alpha (usually the CAPM), a thin
    line marks the spread between the smallest and largest alpha, and the legend
    reports each model's mean absolute alpha and significant count.
    """
    apply_theme()
    models = list(alphas.columns)
    order = alphas[models[0]].sort_values().index
    positions = np.arange(len(order))

    fig, axis = plt.subplots(figsize=(9.5, max(5.0, 0.26 * len(order))))
    for position, asset in zip(positions, order, strict=True):
        low, high = float(alphas.loc[asset].min()), float(alphas.loc[asset].max())
        axis.plot(
            [low, high], [position, position], color=COLORS["muted"], lw=1.0, alpha=0.55, zorder=1
        )
    for index, model in enumerate(models):
        values = alphas.loc[order, model]
        label = model
        if summary is not None and model in summary.index:
            row = summary.loc[model]
            significant = int(row["alphas p<0.05"])
            label = f"{model} (mean |a| {row['mean |alpha|']:.3f}, {significant} sig.)"
        axis.scatter(
            values,
            positions,
            s=46,
            color=_color(model, index),
            edgecolor="white",
            linewidth=0.4,
            label=label,
            zorder=3,
        )
    axis.axvline(0, color="white", lw=1.1)
    axis.set_yticks(positions)
    axis.set_yticklabels(order, fontsize=7)
    axis.set_xlabel("monthly alpha")
    axis.set_title(title)
    axis.legend(loc="best", fontsize=8)
    return _save(fig, output_path)


def plot_factor_cumulative(
    factors: pd.DataFrame,
    output_path: str | Path,
    title: str = "Cumulative self-built factor returns",
) -> Path:
    """Cumulative growth of one dollar in each factor, on each factor's sample."""
    apply_theme()
    fig, axis = plt.subplots(figsize=(9, 5))
    for index, name in enumerate(factors.columns):
        series = factors[name].dropna()
        if series.empty:
            continue
        cumulative = (1.0 + series).cumprod()
        axis.plot(
            cumulative.index,
            cumulative.to_numpy(),
            lw=2.0,
            color=_color(name, index),
            label=name,
        )
    axis.axhline(1.0, lw=0.8, color=COLORS["muted"])
    axis.set_ylabel("cumulative growth of 1 unit")
    axis.set_title(title)
    axis.legend(loc="best")
    return _save(fig, output_path)


# --------------------------------------------------- out-of-sample validation
def plot_oos_growth(
    growth: pd.DataFrame,
    output_path: str | Path,
    title: str = "Out-of-sample growth of 1 unit",
) -> Path:
    """Cumulative growth of each strategy, benchmarks drawn heavier."""
    apply_theme()
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    for index, name in enumerate(growth.columns):
        emphasis = name in OOS_EMPHASIS
        axis.plot(
            growth.index,
            growth[name].to_numpy(),
            lw=2.6 if emphasis else 1.6,
            color=OOS_COLORS.get(name, _color(name, index)),
            alpha=1.0 if emphasis else 0.85,
            label=_label(name, OOS_LABELS),
            zorder=3 if emphasis else 2,
        )
        final = float(growth[name].iloc[-1])
        axis.annotate(
            f"{final:.2f}x",
            (growth.index[-1], final),
            fontsize=8,
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=OOS_COLORS.get(name, COLORS["muted"]),
        )
    axis.axhline(1.0, lw=0.8, color=COLORS["muted"], ls=":")
    axis.set_xlabel("out-of-sample months")
    axis.set_ylabel("growth of 1 unit")
    axis.set_title(title)
    axis.legend(loc="upper left", fontsize=8)
    return _save(fig, output_path)


def plot_oos_sharpe(
    summary: pd.DataFrame,
    output_path: str | Path,
    title: str = "Out-of-sample Sharpe by strategy",
) -> Path:
    """Horizontal bars of annualised Sharpe, with the index as the bar to beat."""
    apply_theme()
    ordered = summary["Sharpe"].sort_values()
    positions = np.arange(len(ordered))
    colors = [OOS_COLORS.get(name, COLORS["long"]) for name in ordered.index]

    fig, axis = plt.subplots(figsize=(9, 4.6))
    axis.barh(positions, ordered.to_numpy(), color=colors, height=0.62, zorder=3)
    for position, (name, value) in zip(positions, ordered.items(), strict=True):
        axis.annotate(
            f"{value:.2f}",
            (value, position),
            fontsize=8,
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=COLORS["muted"],
        )
        if name == "index":
            axis.axvline(value, lw=1.2, ls="--", color=COLORS["accent"], zorder=2)
    axis.set_yticks(positions)
    axis.set_yticklabels([_label(name, OOS_LABELS) for name in ordered.index], fontsize=8)
    axis.set_xlabel("annualised Sharpe ratio (out of sample)")
    axis.set_title(title)
    return _save(fig, output_path)
