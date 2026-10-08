"""Stage 4: Roll's critique — how much does the market proxy matter?

Roll (1977): the CAPM says the *true* market portfolio is mean–variance
efficient, but the true market is unobservable, so every "test of the CAPM" is
really a test of whether the chosen *proxy* is efficient with respect to the
test assets.  Two facts make this concrete:

* the GRS statistic is a function of Sharpe ratios only,
  ``GRS = (T-N-K)/N * (SR*^2 - SR_p^2) / (1 + SR_p^2)``, where ``SR*`` is the
  best Sharpe ratio attainable from the test assets plus the proxy and
  ``SR_p`` the proxy's own — it measures the proxy's distance from the frontier;
* if the proxy *is* the ex-post tangency portfolio of the test assets, every
  alpha is exactly zero and the SML fits perfectly, whatever the economics.

:func:`proxy_comparison` re-runs Stage 2 under several proxies and reports how
betas, alphas, the joint test and the SML move.
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
from .capm import capm_regressions, security_market_line  # noqa: E402
from .factor_models import plot_alpha_comparison  # noqa: E402
from .regression import TimeSeriesFit, grs_test  # noqa: E402

__all__ = [
    "RollResult",
    "ex_post_tangency_proxy",
    "max_sharpe",
    "proxy_comparison",
    "plot_proxy_betas",
    "run_roll_analysis",
]

ANN = config.TRADING_MONTHS
_STYLE = [("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D"), ("#e87ba4", "v")]
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ex_post_tangency_proxy(excess_returns: pd.DataFrame) -> pd.Series:
    """Excess return of the in-sample tangency portfolio of the test assets.

    Used as a "market proxy" it makes every CAPM alpha exactly zero — Roll's
    tautology: a perfect CAPM fit only says the proxy is ex-post efficient.
    """
    mu = excess_returns.mean().to_numpy()
    raw = np.linalg.solve(excess_returns.cov().to_numpy(), mu)
    w = raw / raw.sum()
    return (excess_returns @ w).rename("ex-post tangency")


def max_sharpe(excess_returns: pd.DataFrame) -> float:
    """Highest (monthly) in-sample Sharpe ratio: ``sqrt(mu' Sigma^-1 mu)``, ML moments."""
    x = excess_returns.to_numpy(dtype=float)
    mu = x.mean(axis=0)
    sigma = np.atleast_2d(np.cov(x, rowvar=False, bias=True))
    return float(np.sqrt(mu @ np.linalg.solve(sigma, mu)))


@dataclass
class RollResult:
    """Stage 2 re-run under several market proxies."""

    summary: pd.DataFrame
    betas: pd.DataFrame
    alphas: pd.DataFrame
    fits: dict[str, TimeSeriesFit]
    output_paths: dict[str, Path]


def proxy_comparison(
    excess_returns: pd.DataFrame,
    proxies: dict[str, pd.Series],
    baseline: str | None = None,
    highlight: list[str] | None = None,
) -> RollResult:
    """Run the CAPM regressions and the SML under each proxy.

    ``proxies`` maps a label to a market *excess* return series on the same
    index as ``excess_returns``.  ``baseline`` (default: the first proxy) is
    the reference for the "how much did betas and alphas move" columns.
    ``highlight`` adds one alpha column per named asset.
    """
    names = list(proxies)
    baseline = baseline or names[0]
    fits: dict[str, TimeSeriesFit] = {}
    betas, alphas, rows = {}, {}, {}
    for name in names:
        proxy = proxies[name].reindex(excess_returns.index).rename("market")
        if proxy.isna().any():
            raise ValueError(f"proxy {name!r} does not cover every month.")
        fit = capm_regressions(excess_returns, proxy)
        fits[name] = fit
        betas[name] = fit.params["market"]
        alphas[name] = fit.alpha * ANN

    for name in names:
        fit = fits[name]
        proxy = fit.factors["market"]
        grs = grs_test(fit)
        sml = security_market_line(excess_returns, proxy, betas=betas[name]).tests()
        sr_p = proxy.mean() / proxy.std(ddof=0)
        sr_star = max_sharpe(pd.concat([excess_returns, proxy], axis=1))
        row = {
            "proxy mean (ann.)": proxy.mean() * ANN,
            "proxy sharpe (ann.)": sr_p * np.sqrt(ANN),
            "max sharpe, assets + proxy (ann.)": sr_star * np.sqrt(ANN),
            "GRS": grs.stat,
            "GRS p-value": grs.p_value,
            "mean beta": betas[name].mean(),
            "mean |beta change| vs baseline": (betas[name] - betas[baseline]).abs().mean(),
            # Undefined when a proxy prices everything (alphas are all ~0).
            "corr(alpha, baseline alpha)": (
                alphas[name].corr(alphas[baseline])
                if alphas[name].std() > 1e-10
                else float("nan")
            ),
            "mean |alpha| (ann.)": alphas[name].abs().mean(),
            "|t(alpha)|>2": int((fit.tstats["alpha"].abs() > 2).sum()),
            "SML intercept (ann.)": sml.loc["intercept", "estimate (ann.)"],
            "SML slope (ann.)": sml.loc["slope", "estimate (ann.)"],
            "t(slope=premium)": sml.loc["slope", "t vs theory (Shanken)"],
        }
        for asset in highlight or []:
            row[f"alpha {asset} (ann.)"] = alphas[name][asset]
        rows[name] = row

    summary = pd.DataFrame(rows).T
    summary.index.name = "market proxy"
    beta_frame = pd.DataFrame(betas)
    alpha_frame = pd.DataFrame(alphas)
    beta_frame.index.name = alpha_frame.index.name = "asset"
    return RollResult(summary, beta_frame, alpha_frame, fits, {})


def plot_proxy_betas(
    betas: pd.DataFrame,
    output_path: str | Path,
    baseline: str | None = None,
    title: str = "Betas under alternative market proxies",
) -> Path:
    """Each alternative proxy's beta against the baseline beta, one marker per proxy.

    Points on the 45° line mean the proxy does not matter for that asset.
    """
    output_path = Path(output_path)
    baseline = baseline or betas.columns[0]
    others = [c for c in betas.columns if c != baseline]
    if len(others) > len(_STYLE):
        raise ValueError(f"At most {len(_STYLE)} alternative proxies fit on one chart.")

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.grid(color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    lo = float(min(betas.min().min(), 0.0)) - 0.1
    hi = float(betas.max().max()) + 0.1
    ax.plot([lo, hi], [lo, hi], color=_MUTED, lw=1, ls="--", label="same beta")
    for name, (color, marker) in zip(others, _STYLE, strict=False):
        ax.scatter(
            betas[baseline], betas[name], s=46, marker=marker, color=color,
            edgecolor="white", linewidth=1.0, label=name, zorder=3,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"beta vs {baseline}", color=_INK)
    ax.set_ylabel("beta vs alternative proxy", color=_INK)
    ax.set_title(title, color=_INK, loc="left")
    ax.tick_params(colors=_MUTED)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_roll_analysis(
    excess_returns: pd.DataFrame,
    proxies: dict[str, pd.Series],
    output_dir: str | Path | None = None,
    prefix: str = "",
    baseline: str | None = None,
    highlight: list[str] | None = None,
    chart_proxies: list[str] | None = None,
) -> RollResult:
    """Write the Roll's-critique exhibits: summary, betas, alphas and two charts.

    ``chart_proxies`` (at most four, baseline first) selects the proxies drawn
    on the alpha chart; by default the first four.
    """
    out_dir = config.ensure_outputs_dir(Path(output_dir) if output_dir else None)
    result = proxy_comparison(excess_returns, proxies, baseline, highlight)
    baseline = baseline or list(proxies)[0]
    chart = chart_proxies or list(proxies)[:4]

    paths = {
        "summary": out_dir / f"{prefix}roll_summary.csv",
        "betas": out_dir / f"{prefix}roll_betas.csv",
        "alphas": out_dir / f"{prefix}roll_alphas.csv",
    }
    result.summary.to_csv(paths["summary"])
    result.betas.round(4).to_csv(paths["betas"])
    result.alphas.round(4).to_csv(paths["alphas"])
    paths["beta_chart"] = plot_proxy_betas(
        result.betas[chart], out_dir / f"{prefix}roll_betas.png", baseline=baseline
    )
    paths["alpha_chart"] = plot_alpha_comparison(
        {name: result.fits[name] for name in chart},
        out_dir / f"{prefix}roll_alphas.png",
        title="CAPM alphas under alternative market proxies",
    )
    result.output_paths = paths
    return result
