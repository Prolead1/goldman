"""Stage 3: multi-factor models and what they do to the CAPM alphas.

The market model is augmented with size (SMB), value (HML) and momentum (Mom)
from the Ken French library; the Fama–French five-factor model (adding RMW and
CMA) is a robustness check.  Exhibits:

* (3a) :func:`factor_summary` — premium, t-statistic, volatility and Sharpe of
  each factor over the sample (the construction itself is described in prose);
* (3b) :func:`alpha_comparison`, :func:`model_summary` and
  :func:`plot_alpha_comparison` — alphas before versus after adding factors;
* (3c) :func:`anomaly_verdict` — which CAPM alphas are absorbed, which survive.

A *model* is just a factor frame: ``{"FF3": factors[["Mkt-RF", "SMB", "HML"]]}``.
That keeps the five-factor SMB (built from different sorts) from being mixed up
with the three-factor one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; imported before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402
from .regression import (  # noqa: E402
    TimeSeriesFit,
    chance_exceedances,
    grs_test,
    time_series_regression,
)

__all__ = [
    "MODEL_FACTORS",
    "FactorAnalysisResult",
    "standard_models",
    "factor_summary",
    "fit_models",
    "alpha_comparison",
    "model_summary",
    "anomaly_verdict",
    "factor_loadings",
    "plot_alpha_comparison",
    "run_factor_analysis",
]

ANN = config.TRADING_MONTHS

# The nested model sequence: each adds factors to the one before.
MODEL_FACTORS: dict[str, list[str]] = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "Carhart4": ["Mkt-RF", "SMB", "HML", "Mom"],
}
FF5_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

# Categorical slots 1-3 of the reference palette, one per model, in fixed order,
# paired with distinct marker shapes so identity never rests on colour alone.
_MODEL_STYLE = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#4a3aa7", "D")]
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def standard_models(
    factors: pd.DataFrame,
    index: pd.Index,
    ff5: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """CAPM, FF3 and Carhart four-factor regressors on ``index`` (plus FF5).

    ``factors`` is the output of :func:`goldman.data.load_factors`; ``ff5`` the
    output of :func:`goldman.data.load_ff5`.  Every frame is reindexed to the
    return panel's months and must cover all of them.
    """
    sources: dict[str, tuple[pd.DataFrame, list[str]]] = {
        name: (factors, cols) for name, cols in MODEL_FACTORS.items()
    }
    if ff5 is not None:
        sources["FF5"] = (ff5, FF5_FACTORS)

    models: dict[str, pd.DataFrame] = {}
    for name, (frame, cols) in sources.items():
        aligned = frame.reindex(index)[cols]
        if aligned.isna().any().any():
            raise ValueError(f"{name} factors do not cover every month of the return panel.")
        models[name] = aligned
    return models


# --------------------------------------------------------------- (3a)
def factor_summary(factors: pd.DataFrame) -> pd.DataFrame:
    """Exhibit (3a) companion: how much each factor paid over the sample.

    A factor can only absorb an alpha through its *mean*: ``alpha`` falls by
    ``loading x premium``.  A factor with a near-zero premium raises R² without
    moving alphas, which is the key to reading Stage 3.
    """
    n = factors.notna().sum()
    mean = factors.mean()
    vol = factors.std()
    return pd.DataFrame(
        {
            "mean (ann.)": mean * ANN,
            "t(mean)": mean / (vol / np.sqrt(n)),
            "vol (ann.)": vol * np.sqrt(ANN),
            "sharpe (ann.)": mean / vol * np.sqrt(ANN),
            "months": n,
        }
    )


# --------------------------------------------------------------- (3b)
def fit_models(
    excess_returns: pd.DataFrame,
    models: dict[str, pd.DataFrame],
    hac_lags: int | None = None,
) -> dict[str, TimeSeriesFit]:
    """Run every model's time-series regressions on the same test assets."""
    return {
        name: time_series_regression(excess_returns, regressors, hac_lags)
        for name, regressors in models.items()
    }


def alpha_comparison(fits: dict[str, TimeSeriesFit]) -> pd.DataFrame:
    """Exhibit (3b): annualised alpha, t(alpha) and R² per asset under each model."""
    columns: dict[str, pd.Series] = {}
    for name, fit in fits.items():
        columns[f"{name} alpha (ann.)"] = fit.alpha * ANN
        columns[f"{name} t(alpha)"] = fit.tstats["alpha"]
        columns[f"{name} R2"] = fit.r2
    table = pd.DataFrame(columns)
    table.index.name = "asset"
    return table


def model_summary(fits: dict[str, TimeSeriesFit], threshold: float = 2.0) -> pd.DataFrame:
    """One row per model: the joint test and how big the alphas are overall.

    This is the aggregate version of (3b): if adding factors lowers the GRS
    statistic, the mean absolute alpha and the count of significant alphas
    while raising R², the factors are explaining what the CAPM could not.
    """
    rows = {}
    for name, fit in fits.items():
        grs = grs_test(fit)
        counts = chance_exceedances(fit.tstats["alpha"], fit.df_resid, threshold)
        rows[name] = {
            "factors": fit.factors.shape[1],
            "GRS": grs.stat,
            "GRS p-value": grs.p_value,
            "mean |alpha| (ann.)": float(fit.alpha.abs().mean() * ANN),
            "max |alpha| (ann.)": float(fit.alpha.abs().max() * ANN),
            "|t|>2": counts["observed |t|>2"],
            "expected by chance": counts["expected by chance"],
            "avg R2": float(fit.r2.mean()),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "model"
    return out


# --------------------------------------------------------------- (3c)
def anomaly_verdict(
    fits: dict[str, TimeSeriesFit],
    base: str = "CAPM",
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Exhibit (3c): classify every asset's alpha as the model grows.

    * ``absorbed by <model>`` — significant under ``base``, insignificant from
      the first larger model that explains it;
    * ``survives all models`` — significant under every model;
    * ``emerges under <model>`` — insignificant under ``base`` but significant
      once factors are added (a factor loading exposes a mispricing);
    * ``no anomaly`` — insignificant throughout.

    ``|alpha| change`` compares the richest model with ``base`` (negative means
    the factors shrank the alpha), so near-misses are visible too.
    """
    names = list(fits)
    if base not in fits:
        raise KeyError(f"base model {base!r} not among {names}")
    others = [n for n in names if n != base]
    tstats = pd.DataFrame({n: fits[n].tstats["alpha"] for n in names})
    alphas = pd.DataFrame({n: fits[n].alpha * ANN for n in names})
    significant = tstats.abs() > threshold

    def classify(asset: str) -> str:
        row = significant.loc[asset]
        if row[base]:
            for name in others:
                if not row[name]:
                    return f"absorbed by {name}"
            return "survives all models"
        for name in others:
            if row[name]:
                return f"emerges under {name}"
        return "no anomaly"

    last = others[-1] if others else base
    out = pd.DataFrame(
        {
            f"{base} alpha (ann.)": alphas[base],
            **{f"t {n}": tstats[n] for n in names},
            f"|alpha| change {base}->{last} (ann.)": alphas[last].abs() - alphas[base].abs(),
            "verdict": [classify(a) for a in tstats.index],
        }
    )
    out.index.name = "asset"
    return out


def factor_loadings(fit: TimeSeriesFit) -> pd.DataFrame:
    """Factor loadings with t-statistics, e.g. to see *why* an alpha moved."""
    cols = list(fit.factors.columns)
    table = pd.concat(
        {"loading": fit.params[cols], "t": fit.tstats[cols]}, axis=1
    ).swaplevel(axis=1)
    table = table[[(c, s) for c in cols for s in ("loading", "t")]]
    table.columns = [f"{c} {s}" for c, s in table.columns]
    table["R2"] = fit.r2
    table.index.name = "asset"
    return table


def plot_alpha_comparison(
    fits: dict[str, TimeSeriesFit],
    output_path: str | Path,
    title: str = "Alphas before and after adding factors",
    threshold: float = 2.0,
    order_by: str | None = None,
) -> Path:
    """Exhibit (3b) chart: one row per asset, one marker per model.

    Filled markers are significant (``|t| > threshold``), hollow ones are not,
    so the reader sees both the size of each alpha and whether it is
    distinguishable from zero.  Rows are sorted by the first model's alpha.
    """
    output_path = Path(output_path)
    names = list(fits)
    if len(names) > len(_MODEL_STYLE):
        raise ValueError(f"At most {len(_MODEL_STYLE)} models fit on one chart.")
    order_by = order_by or names[0]
    alphas = pd.DataFrame({n: fits[n].alpha * ANN * 100 for n in names})
    tstats = pd.DataFrame({n: fits[n].tstats["alpha"] for n in names})
    order = alphas[order_by].sort_values().index
    alphas, tstats = alphas.loc[order], tstats.loc[order]

    n_assets = len(order)
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.28 * n_assets + 1.5)))
    ax.grid(axis="x", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axvline(0, color=_MUTED, lw=1)

    rows = np.arange(n_assets)
    span = 0.5
    offsets = np.linspace(-span / 2, span / 2, len(names)) if len(names) > 1 else [0.0]
    for (name, offset), (color, marker) in zip(
        zip(names, offsets, strict=True), _MODEL_STYLE, strict=False
    ):
        sig = tstats[name].abs().to_numpy() > threshold
        y = rows + offset
        x = alphas[name].to_numpy()
        ax.scatter(
            x[sig], y[sig], s=60, marker=marker, color=color, edgecolor=color,
            linewidth=1.6, zorder=3,
        )
        ax.scatter(
            x[~sig], y[~sig], s=46, marker=marker, facecolor="white", edgecolor=color,
            linewidth=1.6, zorder=3,
        )
        ax.scatter([], [], s=46, marker=marker, color=color, label=name)  # legend key

    ax.scatter([], [], s=46, marker="o", color=_MUTED, label=f"filled: |t| > {threshold:g}")
    ax.set_yticks(rows)
    ax.set_yticklabels(order, fontsize=8, color=_INK)
    ax.set_ylim(-0.7, n_assets - 0.3)
    ax.set_xlabel("alpha (% p.a.)", color=_INK)
    ax.set_title(title, color=_INK, loc="left")
    ax.tick_params(colors=_MUTED)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------- one call
@dataclass
class FactorAnalysisResult:
    """Everything Stage 3 produces for one set of test assets."""

    fits: dict[str, TimeSeriesFit]
    comparison: pd.DataFrame
    summary: pd.DataFrame
    verdict: pd.DataFrame
    loadings: pd.DataFrame
    output_paths: dict[str, Path]


def run_factor_analysis(
    excess_returns: pd.DataFrame,
    models: dict[str, pd.DataFrame],
    output_dir: str | Path | None = None,
    prefix: str = "",
    title: str = "Alphas before and after adding factors",
    base: str = "CAPM",
) -> FactorAnalysisResult:
    """Run the Stage 3 exhibits for one set of test assets and write them out.

    Loadings are reported for the last (richest) model in ``models``.
    """
    out_dir = config.ensure_outputs_dir(Path(output_dir) if output_dir else None)

    fits = fit_models(excess_returns, models)
    comparison = alpha_comparison(fits)
    summary = model_summary(fits)
    verdict = anomaly_verdict(fits, base=base)
    loadings = factor_loadings(fits[list(fits)[-1]])

    paths = {
        "comparison": out_dir / f"{prefix}alpha_comparison.csv",
        "summary": out_dir / f"{prefix}model_summary.csv",
        "verdict": out_dir / f"{prefix}anomaly_verdict.csv",
        "loadings": out_dir / f"{prefix}factor_loadings.csv",
    }
    comparison.round(4).to_csv(paths["comparison"])
    summary.to_csv(paths["summary"])
    verdict.round(4).to_csv(paths["verdict"])
    loadings.round(4).to_csv(paths["loadings"])
    paths["chart"] = plot_alpha_comparison(
        fits, out_dir / f"{prefix}alpha_comparison.png", title=title
    )

    return FactorAnalysisResult(
        fits=fits,
        comparison=comparison,
        summary=summary,
        verdict=verdict,
        loadings=loadings,
        output_paths=paths,
    )
