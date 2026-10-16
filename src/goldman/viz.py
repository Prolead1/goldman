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
    "plot_proxy_shift",
    "plot_alpha_comparison",
    "plot_factor_cumulative",
    "plot_oos_growth",
    "plot_oos_sharpe",
    "plot_correlation_paired",
    "plot_risk_return_paired",
    "plot_weights_paired",
    "plot_weight_drift_paired",
    "plot_robustness_paired",
    "plot_alpha_intervals_paired",
    "plot_sml_paired",
    "plot_factor_cumulative_paired",
    "plot_oos_growth_paired",
    "plot_oos_sharpe_paired",
    "plot_index_concentration",
    "plot_custom_factor_growth_paired",
    "plot_custom_factor_scorecard",
    "plot_industry_momentum",
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
            "axes.titleweight": "bold",
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


def _figure_with_axis(figsize, axis):
    """Return ``(figure, axis)``, creating a figure only when no axis is given."""
    if axis is None:
        return plt.subplots(figsize=figsize)
    return axis.figure, axis


def _finish(fig, output_path, axis):
    """Save when an output path is given, otherwise hand the axis back."""
    if output_path is None:
        return axis
    return _save(fig, output_path)


def _label(name: str, mapping: dict[str, str]) -> str:
    return mapping.get(name, name)


def _color(name: str, index: int) -> str:
    return UNIVERSE_COLORS.get(name, list(COLORS.values())[index % len(COLORS)])


# --------------------------------------------------------------- data section
def plot_correlation(
    returns: pd.DataFrame,
    output_path: str | Path | None = None,
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """Correlation heatmap of the universe."""
    apply_theme()
    correlation = returns.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

    fig, axis = _figure_with_axis((9, 7.5), axis)
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
    return _finish(fig, output_path, axis)


# --------------------------------------------------------- Stage 1 portfolio
def plot_risk_return(
    summary: pd.DataFrame,
    output_path: str | Path | None = None,
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """Annualised mean against volatility, coloured by Sharpe."""
    apply_theme()
    fig, axis = _figure_with_axis((9, 6), axis)
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
    return _finish(fig, output_path, axis)


def plot_weights(
    weights: pd.DataFrame,
    output_path: str | Path | None = None,
    comparison_label: str = "comparison",
    title: str = "Estimated tangency weights versus the index",
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """Scatter of index weight against tangency weight, one point per asset."""
    apply_theme()
    index_weight = weights["comparison"]
    tangency = weights["tangency"]
    colors = np.where(tangency >= 0, COLORS["long"], COLORS["short"])

    fig, axis = _figure_with_axis((8.5, 7), axis)
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
    return _finish(fig, output_path, axis)


def plot_weight_drift(
    drift: pd.DataFrame,
    output_path: str | Path | None = None,
    label: str = "",
    title: str = "Tangency weights across subsamples",
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """First-half weights against second-half weights, with the identity line."""
    apply_theme()
    x = drift["first half"]
    y = drift["second half"]
    correlation = float(x.corr(y))

    fig, axis = _figure_with_axis((7, 6.5), axis)
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
    return _finish(fig, output_path, axis)


def plot_robustness(
    weights: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Taming the tangency portfolio",
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """Every asset weight under each construction, as a jittered strip plot."""
    apply_theme()
    generator = np.random.default_rng(0)
    variants = list(weights.columns)
    positions = np.arange(len(variants))

    fig, axis = _figure_with_axis((9.5, 4.8), axis)
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
    return _finish(fig, output_path, axis)


# -------------------------------------------------------------- Stage 2 CAPM

def plot_alpha_intervals(
    table: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Alphas with 95% confidence intervals",
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """Alpha per asset with its confidence interval, coloured by significance."""
    apply_theme()
    ordered = table.sort_values("alpha")
    positions = np.arange(len(ordered))
    significant = ordered["p_alpha"] < 0.05
    colors = np.where(significant, COLORS["short"], COLORS["long"])

    fig, axis = _figure_with_axis((8.5, 8), axis)
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
    return _finish(fig, output_path, axis)


def plot_sml(
    table: pd.DataFrame,
    fit,
    output_path: str | Path | None = None,
    title: str = "Empirical security market line",
    fitted_label: str = "fitted SML",
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """Average excess return against beta, with fitted and theoretical lines."""
    apply_theme()
    beta = table["beta"].to_numpy()
    mean_excess = table["mean_excess"].to_numpy()
    grid = np.linspace(beta.min() - 0.1, beta.max() + 0.1, 50)

    fig, axis = _figure_with_axis((9, 6), axis)
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
    return _finish(fig, output_path, axis)


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


def plot_proxy_shift(
    results: dict,
    output_path: str | Path,
    title: str = "How much moves with the market proxy",
) -> Path:
    """Betas and alphas under one proxy against the other, side by side.

    The companion to :func:`plot_proxy_comparison`: the left panel shows how far
    betas move when the proxy changes and the right panel shows the same for
    alphas, which is the Roll's-critique point that the headline test statistic
    depends on a choice the analyst makes.
    """
    apply_theme()
    methods = list(results)
    first, second = methods[0], methods[1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for axis, field, label in zip(axes, ("beta", "alpha"), ("beta", "monthly alpha"), strict=True):
        x = results[first].table[field]
        y = results[second].table[field]
        axis.scatter(
            x, y, s=60, color=COLORS["long"], edgecolor="white", linewidth=0.4, zorder=3
        )
        low = float(min(x.min(), y.min(), 0.0))
        high = float(max(x.max(), y.max()))
        pad = 0.08 * (high - low)
        limits = [low - pad, high + pad]
        axis.plot(limits, limits, ls="--", lw=1.3, color=COLORS["theory"], label="same value")
        for asset in (y - x).abs().sort_values(ascending=False).head(5).index:
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
        axis.set_xlabel(f"{label}, {_label(first, PROXY_LABELS)}")
        axis.set_ylabel(f"{label}, {_label(second, PROXY_LABELS)}")
        axis.set_title(f"{label.capitalize()} under the two proxies")
        axis.legend(loc="best")
    fig.suptitle(title)
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
    output_path: str | Path | None = None,
    title: str = "Cumulative self-built factor returns",
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """Cumulative growth of one dollar in each factor, on each factor's sample."""
    apply_theme()
    fig, axis = _figure_with_axis((9, 5), axis)
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
    return _finish(fig, output_path, axis)


# --------------------------------------------------- out-of-sample validation
def plot_oos_growth(
    growth: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Out-of-sample growth of 1 unit",
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """Cumulative growth of each strategy, benchmarks drawn heavier."""
    apply_theme()
    fig, axis = _figure_with_axis((9.5, 5.2), axis)
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
    return _finish(fig, output_path, axis)


def plot_oos_sharpe(
    summary: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Out-of-sample Sharpe by strategy",
    axis: plt.Axes | None = None,
) -> Path | plt.Axes:
    """Horizontal bars of annualised Sharpe, with the index as the bar to beat."""
    apply_theme()
    ordered = summary["Sharpe"].sort_values()
    positions = np.arange(len(ordered))
    colors = [OOS_COLORS.get(name, COLORS["long"]) for name in ordered.index]

    fig, axis = _figure_with_axis((9, 4.6), axis)
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
    return _finish(fig, output_path, axis)


# ------------------------------------------------------------- paired panels
def _market_label(name: str) -> str:
    return MARKET_LABELS.get(name, name.upper())


def _paired_panels(mapping, draw, output_path, suptitle, figsize):
    """One row, one panel per entry: draw each onto its own axis and save once."""
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    for axis, (name, data) in zip(axes, mapping.items(), strict=False):
        draw(data, axis)
        axis.set_title(_market_label(name))
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    return _save(fig, output_path)


def plot_correlation_paired(returns_by_universe, output_path: str | Path) -> Path:
    """Correlation heatmaps, one market per panel."""
    return _paired_panels(
        returns_by_universe,
        lambda returns, axis: plot_correlation(returns, axis=axis),
        output_path,
        "Monthly return correlations",
        (14, 6.8),
    )


def plot_risk_return_paired(summaries, output_path: str | Path) -> Path:
    """Risk-return scatters, one market per panel."""
    return _paired_panels(
        summaries,
        lambda summary, axis: plot_risk_return(summary, axis=axis),
        output_path,
        "Risk and return across the universe",
        (14, 6),
    )


def plot_weights_paired(
    weights_by_universe,
    output_path: str | Path,
    comparison_label: str = "value weights",
) -> Path:
    """Tangency-versus-index scatters, one market per panel."""
    return _paired_panels(
        weights_by_universe,
        lambda weights, axis: plot_weights(
            weights, comparison_label=comparison_label, axis=axis
        ),
        output_path,
        "Estimated tangency weights versus the index",
        (14, 6.6),
    )


def plot_weight_drift_paired(drifts, output_path: str | Path) -> Path:
    """Subsample weight scatters, one market per panel."""
    return _paired_panels(
        drifts,
        lambda drift, axis: plot_weight_drift(drift, axis=axis),
        output_path,
        "Tangency weights across subsamples",
        (13, 6.2),
    )


def plot_robustness_paired(
    weights_by_universe,
    diagnostics_by_universe,
    output_path: str | Path,
) -> Path:
    """Weight strip plots under each construction, one market per panel."""
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), layout="constrained")
    for axis, name in zip(axes, weights_by_universe, strict=False):
        plot_robustness(weights_by_universe[name], diagnostics_by_universe[name], axis=axis)
        axis.set_title(_market_label(name))
    fig.suptitle("Taming the tangency portfolio", fontweight="bold")
    return _save(fig, output_path)


def plot_alpha_intervals_paired(tables, output_path: str | Path) -> Path:
    """Alpha confidence-interval charts, one market per panel."""
    return _paired_panels(
        tables,
        lambda table, axis: plot_alpha_intervals(table, axis=axis),
        output_path,
        "Alphas with 95% confidence intervals",
        (15, 8),
    )


def plot_sml_paired(tables, fits, output_path: str | Path) -> Path:
    """Empirical security market lines, one market per panel."""
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    for axis, name in zip(axes, tables, strict=False):
        plot_sml(tables[name], fits[name], axis=axis)
        axis.set_title(_market_label(name))
    fig.suptitle("Empirical security market line", fontweight="bold")
    return _save(fig, output_path)


def plot_factor_cumulative_paired(factors_by_universe, output_path: str | Path) -> Path:
    """Cumulative factor growth, one market per panel."""
    return _paired_panels(
        factors_by_universe,
        lambda factors, axis: plot_factor_cumulative(factors, axis=axis),
        output_path,
        "Cumulative self-built factor returns",
        (13.5, 5),
    )


def plot_oos_growth_paired(growths, output_path: str | Path) -> Path:
    """Out-of-sample growth paths, one market per panel."""
    return _paired_panels(
        growths,
        lambda growth, axis: plot_oos_growth(growth, axis=axis),
        output_path,
        "Out-of-sample growth of 1 unit",
        (14.5, 5.2),
    )


def plot_oos_sharpe_paired(summaries, output_path: str | Path) -> Path:
    """Out-of-sample Sharpe bars, one market per panel."""
    return _paired_panels(
        summaries,
        lambda summary, axis: plot_oos_sharpe(summary, axis=axis),
        output_path,
        "Out-of-sample Sharpe by strategy",
        (14.5, 4.6),
    )


def plot_index_concentration(
    weights_by_universe,
    output_path: str | Path,
    title: str = "The index is a concentrated portfolio, not a broad average",
) -> Path:
    """Two views of value-weight concentration, both markets in one image.

    Left: the concentration curve, the cumulative share of the index held by
    the ``k`` largest names, averaged over the sample.  A straight diagonal is
    what a genuinely broad index would look like; the farther above it the curve
    sits, the more the return depends on a handful of names.  Right: the top-five
    share through time, showing the concentration is persistent rather than a
    single-month accident.
    """
    apply_theme()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5), layout="constrained")

    largest = max(len(weights.columns) for weights in weights_by_universe.values())
    for index, (name, weights) in enumerate(weights_by_universe.items()):
        color = _color(name, index)
        label = _market_label(name)
        average = weights.mean().sort_values(ascending=False)
        cumulative = average.cumsum() / average.sum()
        ranks = np.arange(1, len(cumulative) + 1)
        left.plot(ranks, cumulative.to_numpy(), marker="o", ms=3, lw=2.0, color=color, label=label)
        top_five = float(average.nlargest(5).sum() / average.sum())
        left.annotate(
            f"{label}: top five = {top_five:.0%}",
            (5, float(cumulative.iloc[4])),
            xytext=(12, 9 if index == 0 else -20),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
        top_five_series = weights.apply(lambda row: row.nlargest(5).sum(), axis=1)
        right.plot(
            top_five_series.index, top_five_series.to_numpy(), lw=1.8, color=color, label=label
        )

    left.plot(
        [1, largest],
        [1.0 / largest, 1.0],
        ls="--",
        lw=1.2,
        color=COLORS["muted"],
        label="equal weight",
    )
    left.axvline(5, ls=":", lw=1.1, color=COLORS["accent"])
    left.set_xlabel("number of holdings, largest first")
    left.set_ylabel("cumulative share of the index")
    left.set_title("Concentration curve")
    left.legend(loc="lower right", fontsize=8)

    right.axhline(0.5, ls="--", lw=1.0, color=COLORS["muted"])
    right.set_ylim(0.0, 1.0)
    right.set_ylabel("top-five share of the index")
    right.set_title("Top-five weight through time")
    right.legend(loc="lower right", fontsize=8)

    fig.suptitle(title, fontweight="bold")
    return _save(fig, output_path)


# ------------------------------------------------------------ custom factors
#: Momentum is the reference, the two refinements the challengers, the market
#: the backdrop.
CUSTOM_COLORS = {
    "WML": COLORS["muted"],
    "NEAR52": COLORS["accent"],
    "VWML": COLORS["long"],
    "Mkt-RF": COLORS["theory"],
}
CUSTOM_LABELS = {
    "WML": "momentum (WML)",
    "NEAR52": "52-week high (NEAR52)",
    "VWML": "vol-managed momentum (VWML)",
    "Mkt-RF": "market excess return",
}


def _plot_custom_growth(panel: pd.DataFrame, axis: plt.Axes) -> plt.Axes:
    """Growth of 1 unit in each column of ``panel`` on its common sample."""
    growth = (1.0 + panel.dropna()).cumprod()
    for name in growth.columns:
        is_market = name == "Mkt-RF"
        axis.plot(
            growth.index,
            growth[name].to_numpy(),
            lw=1.4 if is_market else (2.6 if name != "WML" else 1.8),
            ls="--" if is_market else "-",
            color=CUSTOM_COLORS.get(name, COLORS["muted"]),
            label=CUSTOM_LABELS.get(name, name),
            zorder=2 if is_market or name == "WML" else 3,
        )
        final = float(growth[name].iloc[-1])
        axis.annotate(
            f"{final:.2f}x",
            (growth.index[-1], final),
            fontsize=8,
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=CUSTOM_COLORS.get(name, COLORS["muted"]),
        )
    axis.axhline(1.0, lw=0.8, color=COLORS["muted"], ls=":")
    axis.set_ylabel("growth of 1 unit")
    axis.legend(loc="upper left", fontsize=8)
    return axis


def plot_custom_factor_growth_paired(panels, output_path: str | Path) -> Path:
    """Cumulative growth of the custom factors against momentum and the market.

    ``panels`` maps a universe to a frame with columns such as ``WML``,
    ``NEAR52``, ``VWML`` and ``Mkt-RF``; each panel uses the months where all of
    them exist, so the lines start together.
    """
    return _paired_panels(
        panels,
        _plot_custom_growth,
        output_path,
        "Two refinements of momentum, the same rules in both markets",
        (14.5, 5.2),
    )


def plot_custom_factor_scorecard(
    scorecards,
    output_path: str | Path,
    factors=("WML", "NEAR52", "VWML"),
) -> Path:
    """Sharpe ratio and CAPM alpha with 95% intervals, both markets on one axis.

    ``scorecards`` maps a universe to the table from
    :func:`goldman.analysis.custom_factors.factor_scorecard`.  Each factor gets
    one row per market, so a reader can see in one glance which factor works
    where and whether its interval clears zero.
    """
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), layout="constrained")
    panels = [
        ("Sharpe", "Sharpe se", "annualised Sharpe ratio", "Sharpe ratio"),
        ("CAPM alpha (ann.)", None, "annualised CAPM alpha", "CAPM alpha"),
    ]
    rows = np.arange(len(factors))[::-1]
    offsets = np.linspace(0.15, -0.15, len(scorecards)) if len(scorecards) > 1 else [0.0]
    for axis, (column, se_column, xlabel, title) in zip(axes, panels, strict=True):
        for offset, (name, table) in zip(offsets, scorecards.items(), strict=True):
            values = table.loc[list(factors), column].astype(float).to_numpy()
            if se_column is None:
                se = np.abs(values / table.loc[list(factors), "t(CAPM alpha)"].astype(float))
            else:
                se = table.loc[list(factors), se_column].astype(float).to_numpy()
            axis.errorbar(
                values,
                rows + offset,
                xerr=1.96 * np.asarray(se),
                fmt="o",
                ms=7,
                capsize=3,
                lw=1.6,
                color=_color(name, 0),
                label=_market_label(name),
            )
        axis.axvline(0.0, color="white", lw=1.0)
        axis.set_yticks(rows)
        axis.set_yticklabels([CUSTOM_LABELS.get(f, f) for f in factors], fontsize=9)
        axis.set_xlabel(xlabel)
        axis.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=9)
    fig.suptitle("Which factor works where (bars are 95% intervals)", fontweight="bold")
    return _save(fig, output_path)


# ---------------------------------------------------------- industry momentum
INDUSTRY_COLORS = {"IndMom": COLORS["accent"], "Mom": COLORS["long"], "Mkt-RF": COLORS["theory"]}
INDUSTRY_LABELS = {
    "IndMom": "industry momentum (ours)",
    "Mom": "Ken French stock momentum",
    "Mkt-RF": "market excess return",
}


def plot_industry_momentum(
    panel: pd.DataFrame,
    output_path: str | Path,
    *,
    window: int = 120,
    highlight: tuple[str, str] | None = None,
) -> Path:
    """Sixty years of momentum: growth of 1 unit and the rolling t-statistic.

    ``panel`` holds monthly ``IndMom``, ``Mom`` and ``Mkt-RF`` on a
    ``DatetimeIndex``.  The left panel compounds each on a log scale; the right
    plots the t-statistic of each factor's mean over the trailing ``window``
    months, the evidence a ten-year sample like ours would give.  ``highlight``
    shades the months our own data cover.
    """
    apply_theme()
    panel = panel.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), layout="constrained")

    growth = (1.0 + panel).cumprod()
    finals = growth.iloc[-1].sort_values()
    # Push end labels apart when two lines finish close together on the log axis.
    nudge = dict.fromkeys(finals.index, 0)
    for low, high in zip(finals.index[:-1], finals.index[1:], strict=True):
        if np.log10(finals[high] / finals[low]) < 0.08:
            nudge[low], nudge[high] = nudge[low] - 6, nudge[high] + 6
    for name in growth.columns:
        axes[0].plot(
            growth.index,
            growth[name].to_numpy(),
            lw=2.4 if name == "IndMom" else 1.5,
            ls="--" if name == "Mkt-RF" else "-",
            color=INDUSTRY_COLORS.get(name, COLORS["muted"]),
            label=INDUSTRY_LABELS.get(name, name),
        )
        axes[0].annotate(
            f"{finals[name]:.1f}x",
            (growth.index[-1], float(finals[name])),
            fontsize=8,
            xytext=(5, nudge[name]),
            textcoords="offset points",
            va="center",
            color=INDUSTRY_COLORS.get(name, COLORS["muted"]),
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("growth of 1 unit (log scale)")
    axes[0].set_title(f"Growth of 1 unit, {panel.index[0]:%Y-%m} to {panel.index[-1]:%Y-%m}")
    axes[0].legend(loc="upper left", fontsize=8)

    for name in ("IndMom", "Mom"):
        series = panel[name]
        t_stat = series.rolling(window).mean() / series.rolling(window).std() * np.sqrt(window)
        axes[1].plot(
            t_stat.index,
            t_stat.to_numpy(),
            lw=2.2 if name == "IndMom" else 1.5,
            color=INDUSTRY_COLORS[name],
            label=INDUSTRY_LABELS[name],
        )
    axes[1].axhline(1.96, color=COLORS["short"], lw=1.0, ls="--", label="t = 1.96")
    axes[1].axhline(0.0, color="white", lw=0.8)
    axes[1].set_ylabel(f"t-statistic of the mean, trailing {window // 12} years")
    axes[1].set_title(f"What a {window // 12}-year sample would have said")
    axes[1].legend(loc="upper right", fontsize=8)

    title = "Industry momentum on 10 Ken French industries"
    if highlight is not None:
        start, end = pd.Timestamp(highlight[0]), pd.Timestamp(highlight[1])
        for axis in axes:
            axis.axvspan(start, end, color=COLORS["muted"], alpha=0.18, lw=0)
        title += " (shaded: the decade our data cover)"
    fig.suptitle(title, fontweight="bold")
    return _save(fig, output_path)
