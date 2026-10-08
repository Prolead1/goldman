"""Stage 2: the CAPM and the empirical security market line.

For every asset the security characteristic line

    R_i,t - R_f,t = alpha_i + beta_i (R_M,t - R_f,t) + e_i,t

is estimated by OLS.  The exhibits are:

* (2a) :func:`capm_table` — beta, alpha, t(alpha) and R² per asset;
* (2b) :func:`plot_sml` — average excess return against beta, with the fitted
  line and the theoretical line ``beta * (mean R_M - R_f)`` through the origin;
* (2c) :func:`alpha_joint_test` — the GRS test plus the "how many ``|t| > 2``
  would chance give" count;
* (2d) :func:`security_market_line` — Fama–MacBeth slope and intercept tested
  against the theory (intercept 0, slope = market premium).

The market proxy is an argument everywhere, so re-running the stage with a
different proxy (Roll's critique) is one extra call.
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
    FamaMacBethResult,
    GRSResult,
    TimeSeriesFit,
    chance_exceedances,
    fama_macbeth,
    grs_test,
    time_series_regression,
)

__all__ = [
    "SMLResult",
    "CapmAnalysisResult",
    "capm_regressions",
    "capm_table",
    "alpha_joint_test",
    "security_market_line",
    "sml_comparison",
    "plot_sml",
    "run_capm_analysis",
]

ANN = config.TRADING_MONTHS

# Chart colours: categorical slots 1-2 of the reference palette, neutral ink.
_BLUE, _ORANGE = "#2a78d6", "#eb6834"
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _as_market_frame(market_excess: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(market_excess, pd.DataFrame):
        if market_excess.shape[1] != 1:
            raise ValueError("market_excess must be a single series.")
        return market_excess
    return market_excess.to_frame(market_excess.name or "Mkt-RF")


# --------------------------------------------------------------- (2a)
def capm_regressions(
    excess_returns: pd.DataFrame,
    market_excess: pd.Series | pd.DataFrame,
    hac_lags: int | None = None,
) -> TimeSeriesFit:
    """Run the security characteristic regression for every asset."""
    return time_series_regression(excess_returns, _as_market_frame(market_excess), hac_lags)


def capm_table(fit: TimeSeriesFit) -> pd.DataFrame:
    """Exhibit (2a): beta, annualised alpha, t-statistics and R² per asset.

    ``t(alpha)`` uses homoskedastic OLS errors; ``t(alpha) NW`` uses Newey–West
    errors (robust to heteroskedasticity and autocorrelation) as a check that
    the inference does not hinge on the error assumptions.
    """
    market = fit.factors.columns[0]
    table = pd.DataFrame(
        {
            "beta": fit.params[market],
            "t(beta)": fit.tstats[market],
            "alpha (ann.)": fit.alpha * ANN,
            "t(alpha)": fit.tstats["alpha"],
            "t(alpha) NW": fit.tstats_hac["alpha"],
            "R2": fit.r2,
            "resid vol (ann.)": fit.resid.std(ddof=fit.factors.shape[1] + 1) * np.sqrt(ANN),
        }
    )
    table.index.name = "asset"
    return table


# --------------------------------------------------------------- (2c)
def alpha_joint_test(fit: TimeSeriesFit, threshold: float = 2.0) -> pd.Series:
    """Exhibit (2c): is "all alphas are zero" rejected?

    Combines the GRS F-test with the informal count of individually significant
    alphas against what chance alone would produce.  Residual correlations are
    reported because they are why the count is only indicative (the tests are
    not independent) and the GRS test is the formal answer.  The mean
    *absolute* correlation matters: positive and negative pairs (e.g. sector
    rotations) cancel in the plain mean.
    """
    grs = grs_test(fit)
    counts = chance_exceedances(fit.tstats["alpha"], fit.df_resid, threshold)
    corr = fit.resid.corr().to_numpy()
    off_diag = corr[~np.eye(len(corr), dtype=bool)]
    out = {
        "GRS statistic": grs.stat,
        "GRS p-value": grs.p_value,
        "GRS df": f"F({grs.df_num}, {grs.df_den})",
        **counts,
        "observed |t|>2 (NW)": int((fit.tstats_hac["alpha"].abs() > threshold).sum()),
        "mean |alpha| (ann.)": float(fit.alpha.abs().mean() * ANN),
        # Smallest alpha a 5% two-sided test detects with 80% power:
        # (1.96 + 0.84) x se(alpha), for the median asset.
        "min detectable alpha (ann., median asset)": float(
            2.8016 * (fit.alpha / fit.tstats["alpha"]).abs().median() * ANN
        ),
        "avg residual correlation": float(off_diag.mean()),
        "avg |residual correlation|": float(np.abs(off_diag).mean()),
    }
    return pd.Series(out, name="joint test")


# --------------------------------------------------------------- (2d)
@dataclass
class SMLResult:
    """The empirical SML and its comparison with the CAPM's prediction.

    All stored values are monthly; :meth:`tests` reports them annualised.
    """

    betas: pd.Series
    mean_excess: pd.Series
    market_premium: float
    fama_macbeth: FamaMacBethResult

    @property
    def intercept(self) -> float:
        return float(self.fama_macbeth.gamma.iloc[0])

    @property
    def slope(self) -> float:
        return float(self.fama_macbeth.gamma.iloc[1])

    def tests(self) -> pd.DataFrame:
        """Fitted versus theoretical intercept and slope, with t-statistics.

        Theory: intercept 0 and slope equal to the average market excess
        return.  ``t vs theory`` is ``(estimate - theory) / se``; the Shanken
        column corrects the standard error for betas being estimated.
        """
        fm = self.fama_macbeth
        estimate = np.array([self.intercept, self.slope])
        theory = np.array([0.0, self.market_premium])
        se = fm.se.to_numpy()
        se_s = fm.se_shanken.to_numpy()
        return pd.DataFrame(
            {
                "estimate (ann.)": estimate * ANN,
                "theory (ann.)": theory * ANN,
                "se FM (ann.)": se * ANN,
                "t vs theory (FM)": (estimate - theory) / se,
                "t vs theory (Shanken)": (estimate - theory) / se_s,
            },
            index=pd.Index(["intercept", "slope"], name="SML"),
        )


def security_market_line(
    excess_returns: pd.DataFrame,
    market_excess: pd.Series | pd.DataFrame,
    betas: pd.Series | None = None,
) -> SMLResult:
    """Fit the empirical SML: average excess return on beta (Fama–MacBeth).

    ``betas`` default to the full-sample time-series estimates.  The slope and
    intercept are the means of the monthly cross-sectional regressions, whose
    standard errors respect the cross-sectional correlation of returns.
    """
    market = _as_market_frame(market_excess)
    if betas is None:
        fit = capm_regressions(excess_returns, market, hac_lags=0)
        betas = fit.params[market.columns[0]]
    fm = fama_macbeth(excess_returns, betas.rename("beta"), market)
    return SMLResult(
        betas=betas.rename("beta"),
        mean_excess=excess_returns.mean(),
        market_premium=float(market.iloc[:, 0].mean()),
        fama_macbeth=fm,
    )


def sml_comparison(
    cases: dict[str, tuple[pd.DataFrame, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Fit the SML on several test-asset sets or periods, one row each.

    ``cases`` maps a label to ``(excess_returns, market_excess)``.  This is the
    (2d) robustness table: does the slope/intercept verdict survive dropping
    an outlier, changing the test assets, or splitting the sample?
    """
    rows = {}
    for name, (excess, market) in cases.items():
        sml = security_market_line(excess, market)
        tests = sml.tests()
        rows[name] = {
            "assets": excess.shape[1],
            "months": len(excess),
            "intercept (ann.)": tests.loc["intercept", "estimate (ann.)"],
            "t(intercept=0)": tests.loc["intercept", "t vs theory (Shanken)"],
            "slope (ann.)": tests.loc["slope", "estimate (ann.)"],
            "market premium (ann.)": tests.loc["slope", "theory (ann.)"],
            "t(slope=premium)": tests.loc["slope", "t vs theory (Shanken)"],
            "cross-sectional R2": sml.fama_macbeth.r2_cross_section,
        }
    out = pd.DataFrame(rows).T
    out.index.name = "case"
    return out


# --------------------------------------------------------------- (2b)
def _auto_labels(sml: SMLResult, n: int) -> list[str]:
    """The points furthest from the theoretical line, plus the beta extremes."""
    gap = (sml.mean_excess - sml.betas * sml.market_premium).abs()
    picks = list(gap.nlargest(n).index)
    picks += [sml.betas.idxmax(), sml.betas.idxmin()]
    return list(dict.fromkeys(picks))


def plot_sml(
    sml: SMLResult,
    output_path: str | Path,
    title: str = "Empirical security market line",
    labels: str | list[str] = "auto",
    n_labels: int = 8,
) -> Path:
    """Exhibit (2b): mean excess return against beta, fitted vs theoretical SML.

    ``labels`` is ``"auto"`` (largest deviations from theory and the beta
    extremes), ``"all"``, ``"none"`` or an explicit list of asset names.
    """
    output_path = Path(output_path)
    betas = sml.betas
    means = sml.mean_excess.loc[betas.index] * ANN * 100
    premium = sml.market_premium * ANN * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid(color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    lo = min(0.0, float(betas.min()) - 0.1)
    hi = float(betas.max()) + 0.15
    grid = np.linspace(lo, hi, 50)
    ax.plot(grid, premium * grid, color=_ORANGE, lw=2, ls="--", label="theoretical SML")
    fitted = (sml.intercept + sml.slope * grid) * ANN * 100
    ax.plot(grid, fitted, color=_BLUE, lw=2, label="fitted SML (Fama–MacBeth)")
    ax.scatter(
        betas, means, s=48, color=_BLUE, edgecolor="white", linewidth=1.5, zorder=3, label="assets"
    )
    ax.scatter(
        [1.0], [premium], s=110, marker="D", color=_ORANGE, edgecolor="white",
        linewidth=1.5, zorder=4, label="market proxy",
    )
    ax.axhline(0, color=_MUTED, lw=0.8)

    if labels == "all":
        to_label = list(betas.index)
    elif labels == "auto":
        to_label = _auto_labels(sml, n_labels)
    elif labels == "none":
        to_label = []
    else:
        to_label = list(labels)
    for name in to_label:
        ax.annotate(
            str(name), (betas[name], means[name]), xytext=(5, 4),
            textcoords="offset points", fontsize=8, color=_MUTED,
        )

    tests = sml.tests()
    note = (
        f"fitted: {tests.loc['intercept', 'estimate (ann.)'] * 100:+.1f}% "
        f"+ {tests.loc['slope', 'estimate (ann.)'] * 100:.1f}% x beta\n"
        f"theory:  0.0% + {premium:.1f}% x beta"
    )
    ax.text(
        0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=9, color=_INK,
        family="monospace",
    )
    ax.set_xlabel(r"CAPM beta $\hat\beta_i$", color=_INK)
    ax.set_ylabel("average excess return (% p.a.)", color=_INK)
    ax.set_title(title, color=_INK, loc="left")
    ax.tick_params(colors=_MUTED)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, fontsize=9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------- one call
@dataclass
class CapmAnalysisResult:
    """Everything Stage 2 produces for one set of test assets."""

    fit: TimeSeriesFit
    table: pd.DataFrame
    joint_test: pd.Series
    grs: GRSResult
    sml: SMLResult
    output_paths: dict[str, Path]


def run_capm_analysis(
    excess_returns: pd.DataFrame,
    market_excess: pd.Series | pd.DataFrame,
    output_dir: str | Path | None = None,
    prefix: str = "",
    title: str = "Empirical security market line",
    labels: str | list[str] = "auto",
) -> CapmAnalysisResult:
    """Run every Stage 2 exhibit and write them to ``output_dir``.

    ``prefix`` distinguishes several runs in one directory (e.g. ``"ff25_"``
    for the sorted portfolios or ``"msci_"`` for another market proxy).
    """
    out_dir = config.ensure_outputs_dir(Path(output_dir) if output_dir else None)

    fit = capm_regressions(excess_returns, market_excess)
    table = capm_table(fit)
    joint = alpha_joint_test(fit)
    sml = security_market_line(excess_returns, market_excess, betas=table["beta"])

    paths = {
        "table": out_dir / f"{prefix}capm_table.csv",
        "joint_test": out_dir / f"{prefix}alpha_joint_test.csv",
        "sml_tests": out_dir / f"{prefix}sml_tests.csv",
    }
    table.round(4).to_csv(paths["table"])
    joint.to_csv(paths["joint_test"])
    sml.tests().round(4).to_csv(paths["sml_tests"])
    paths["sml"] = plot_sml(sml, out_dir / f"{prefix}sml.png", title=title, labels=labels)

    return CapmAnalysisResult(
        fit=fit,
        table=table,
        joint_test=joint,
        grs=grs_test(fit),
        sml=sml,
        output_paths=paths,
    )
