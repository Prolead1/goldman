"""Smoke tests for the visualisation module: every figure must render."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from goldman import viz
from goldman.analysis import (
    Tradability,
    empirical_sml,
    run_robustness_analysis,
    security_characteristic_line,
    subsample_tangency,
    summary_statistics,
    tangency_vs_comparison,
)


def _panel(n: int = 120, k: int = 8, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2010-01-01", periods=n, freq="MS")
    market = pd.Series(rng.normal(0.006, 0.04, n), index=index)
    data = {
        f"a{i}": 0.002 + (0.5 + 0.2 * i) * market.to_numpy() + rng.normal(0.0, 0.02, n)
        for i in range(k)
    }
    return pd.DataFrame(data, index=index), market


def _assert_png(path) -> None:
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 1000


def test_plot_correlation(tmp_path):
    returns, _ = _panel()
    _assert_png(viz.plot_correlation(returns, tmp_path / "corr.png"))


def test_plot_risk_return(tmp_path):
    returns, _ = _panel()
    summary = summary_statistics(returns, rf=0.003)
    _assert_png(viz.plot_risk_return(summary, tmp_path / "risk.png"))


def test_plot_weights(tmp_path):
    returns, _ = _panel()
    weights = tangency_vs_comparison(returns, rf=0.003)
    _assert_png(viz.plot_weights(weights, tmp_path / "weights.png"))


def test_plot_weight_drift(tmp_path):
    returns, _ = _panel()
    drift = subsample_tangency(returns, rf=0.003)
    _assert_png(viz.plot_weight_drift(drift, tmp_path / "drift.png", label="us"))


def test_plot_robustness(tmp_path):
    returns, _ = _panel()
    result = run_robustness_analysis(returns, rf=0.003, output_dir=tmp_path)
    _assert_png(
        viz.plot_robustness(result.weights, result.diagnostics, tmp_path / "robustness.png")
    )
    # run_robustness_analysis also renders the chart through viz.
    _assert_png(result.output_paths["chart"])


def test_plot_alpha_intervals(tmp_path):
    returns, market = _panel()
    table = security_characteristic_line(returns, market, rf=0.0)
    _assert_png(viz.plot_alpha_intervals(table, tmp_path / "alpha.png"))


def test_plot_sml(tmp_path):
    returns, market = _panel()
    table = security_characteristic_line(returns, market, rf=0.0)
    fit = empirical_sml(table, market)
    _assert_png(viz.plot_sml(table, fit, tmp_path / "sml.png"))


def test_plot_grs(tmp_path):
    results = {
        "us": SimpleNamespace(
            table=pd.DataFrame({"t_alpha": [-2.1, -1.0, 0.5, 2.4, 0.1]}), grs={"p_value": 0.47}
        ),
        "sg": SimpleNamespace(
            table=pd.DataFrame({"t_alpha": [-2.6, -1.5, 0.2, 1.9, 2.2, 0.7]}),
            grs={"p_value": 0.09},
        ),
    }
    _assert_png(viz.plot_grs(results, tmp_path / "grs.png"))


def test_plot_sml_fit(tmp_path):
    def fit(slope: float, theory: float, intercept: float):
        return SimpleNamespace(slope=slope, theoretical_slope=theory, intercept=intercept)

    results = {
        "us": SimpleNamespace(fit=fit(0.016, 0.019, 0.0004)),
        "sg": SimpleNamespace(fit=fit(0.018, 0.014, -0.006)),
    }
    _assert_png(viz.plot_sml_fit(results, tmp_path / "fit.png"))


def test_plot_capm_diagnostics(tmp_path):
    def result(slope, theory, intercept, t_stats, p_value):
        return SimpleNamespace(
            fit=SimpleNamespace(slope=slope, theoretical_slope=theory, intercept=intercept),
            table=pd.DataFrame({"t_alpha": t_stats}),
            grs={"p_value": p_value},
        )

    results = {
        "us": result(0.016, 0.019, 0.0004, [-2.1, -1.0, 0.5, 2.4, 0.1], 0.47),
        "sg": result(0.018, 0.014, -0.006, [-2.6, -1.5, 0.2, 1.9, 2.2], 0.09),
    }
    _assert_png(viz.plot_capm_diagnostics(results, tmp_path / "capm_diagnostics.png"))


def test_plot_proxy_comparison(tmp_path):
    index = ["a", "b", "c"]
    results = {
        "value_weighted": SimpleNamespace(
            table=pd.DataFrame({"beta": [0.8, 1.1, 1.5]}, index=index),
            grs={"p_value": 0.47},
        ),
        "ff_market": SimpleNamespace(
            table=pd.DataFrame({"beta": [0.9, 1.0, 1.4]}, index=index),
            grs={"p_value": 0.19},
        ),
    }
    _assert_png(viz.plot_proxy_comparison(results, tmp_path / "proxy.png"))


def test_plot_proxy_shift(tmp_path):
    index = ["a", "b", "c"]
    results = {
        "value_weighted": SimpleNamespace(
            table=pd.DataFrame(
                {"beta": [0.8, 1.1, 1.5], "alpha": [0.002, -0.004, 0.001]}, index=index
            )
        ),
        "ff_market": SimpleNamespace(
            table=pd.DataFrame(
                {"beta": [0.9, 1.0, 1.4], "alpha": [-0.001, -0.002, 0.003]}, index=index
            )
        ),
    }
    _assert_png(viz.plot_proxy_shift(results, tmp_path / "proxy_shift.png"))


def test_plot_alpha_comparison(tmp_path):
    index = list("abcd")
    alphas = pd.DataFrame(
        {"CAPM": [0.010, -0.004, 0.002, 0.006], "Two-factor": [0.002, -0.001, 0.001, -0.003]},
        index=index,
    )
    summary = pd.DataFrame(
        {
            "mean |alpha|": [0.0055, 0.00175],
            "alphas p<0.05": [1, 0],
        },
        index=["CAPM", "Two-factor"],
    )
    _assert_png(viz.plot_alpha_comparison(alphas, summary, tmp_path / "alphas.png"))


def test_plot_alpha_comparison_paired(tmp_path):
    index = list("abcd")
    alphas = pd.DataFrame(
        {"CAPM": [0.010, -0.004, 0.002, 0.006], "Two-factor": [0.002, -0.001, 0.001, -0.003]},
        index=index,
    )
    summary = pd.DataFrame(
        {"mean |alpha|": [0.0055, 0.00175], "alphas p<0.05": [1, 0]},
        index=["CAPM", "Two-factor"],
    )
    results = {
        "us": SimpleNamespace(alphas=alphas, summary=summary),
        "sg": SimpleNamespace(alphas=alphas * 0.5, summary=summary),
    }
    _assert_png(viz.plot_alpha_comparison_paired(results, tmp_path / "alphas_paired.png"))


def test_plot_factor_cumulative(tmp_path):
    index = pd.date_range("2010-01-01", periods=24, freq="MS")
    factors = pd.DataFrame(
        {
            "SMB": np.linspace(-0.02, 0.03, 24),
            "WML": np.r_[np.full(6, np.nan), np.linspace(-0.01, 0.04, 18)],
        },
        index=index,
    )
    _assert_png(viz.plot_factor_cumulative(factors, tmp_path / "factors.png"))


def test_plot_oos_growth(tmp_path):
    index = pd.date_range("2010-01-01", periods=24, freq="MS")
    growth = pd.DataFrame(
        {
            "index": np.linspace(1.0, 1.40, 24),
            "equal_weight": np.linspace(1.0, 1.20, 24),
            "unconstrained": np.linspace(1.0, 0.90, 24),
        },
        index=index,
    )
    _assert_png(viz.plot_oos_growth(growth, tmp_path / "growth.png"))


def test_plot_oos_sharpe(tmp_path):
    summary = pd.DataFrame(
        {"Sharpe": [1.10, 0.90, 0.45]},
        index=["index", "equal_weight", "unconstrained"],
    )
    _assert_png(viz.plot_oos_sharpe(summary, tmp_path / "sharpe.png"))


# ------------------------------------------------------- paired-market panels
def _two_panels():
    us, market_us = _panel(seed=0)
    sg, market_sg = _panel(seed=1)
    return {"us": us, "sg": sg}, {"us": market_us, "sg": market_sg}


def test_plot_paired_portfolio_charts(tmp_path):
    returns, markets = _two_panels()
    summaries = {u: summary_statistics(r, rf=0.003) for u, r in returns.items()}
    weights = {u: tangency_vs_comparison(r, rf=0.003) for u, r in returns.items()}
    drifts = {u: subsample_tangency(r, rf=0.003) for u, r in returns.items()}
    _assert_png(
        viz.plot_correlation_paired(returns, tmp_path / "corr_paired.png")
    )
    _assert_png(viz.plot_risk_return_paired(summaries, tmp_path / "risk_paired.png"))
    _assert_png(viz.plot_weights_paired(weights, tmp_path / "weights_paired.png"))
    _assert_png(viz.plot_weight_drift_paired(drifts, tmp_path / "drift_paired.png"))
    assert set(markets) == set(returns)


def test_plot_paired_robustness(tmp_path):
    returns, _ = _two_panels()
    fixes = {
        u: run_robustness_analysis(r, rf=0.003, output_dir=tmp_path / u, label=u)
        for u, r in returns.items()
    }
    _assert_png(
        viz.plot_robustness_paired(
            {u: fixes[u].weights for u in fixes},
            {u: fixes[u].diagnostics for u in fixes},
            tmp_path / "robustness_paired.png",
        )
    )


def test_plot_paired_capm_charts(tmp_path):
    returns, markets = _two_panels()
    tables = {
        u: security_characteristic_line(returns[u], markets[u], rf=0.0) for u in returns
    }
    fits = {u: empirical_sml(tables[u], markets[u]) for u in returns}
    _assert_png(viz.plot_alpha_intervals_paired(tables, tmp_path / "alpha_paired.png"))
    _assert_png(viz.plot_sml_paired(tables, fits, tmp_path / "sml_paired.png"))


def test_plot_paired_factor_and_oos(tmp_path):
    index = pd.date_range("2010-01-01", periods=24, freq="MS")
    factors = {
        u: pd.DataFrame(
            {"SMB": np.linspace(-0.02, 0.03, 24), "WML": np.linspace(-0.01, 0.04, 24)},
            index=index,
        )
        for u in ("us", "sg")
    }
    growths = {
        u: pd.DataFrame(
            {"index": np.linspace(1.0, 1.4, 24), "equal_weight": np.linspace(1.0, 1.2, 24)},
            index=index,
        )
        for u in ("us", "sg")
    }
    summaries = {
        u: pd.DataFrame({"Sharpe": [1.10, 0.90]}, index=["index", "equal_weight"])
        for u in ("us", "sg")
    }
    _assert_png(viz.plot_factor_cumulative_paired(factors, tmp_path / "factors_paired.png"))
    _assert_png(viz.plot_oos_growth_paired(growths, tmp_path / "growth_paired.png"))
    _assert_png(viz.plot_oos_sharpe_paired(summaries, tmp_path / "sharpe_paired.png"))


def test_plot_factor_tradability(tmp_path):
    index = pd.date_range("2010-01-01", periods=24, freq="MS")

    def result(scale: float) -> Tradability:
        factor = pd.Series(np.linspace(-0.03, 0.03, 24) * scale, index=index)
        turnover = pd.Series(0.5, index=index)
        wealth = (1.0 + factor).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        return Tradability(
            factor=factor,
            turnover=turnover,
            drawdown=drawdown,
            break_even_bps=30.0 * scale,
        )

    scales = (("SMB", 1.0), ("WML", 2.0), ("LOWVOL", 1.5), ("REV", 0.8))
    panels = {
        "us": {name: result(scale) for name, scale in scales},
        "sg": {name: result(scale * 0.5) for name, scale in scales},
    }
    _assert_png(viz.plot_factor_tradability(panels, tmp_path / "factor_tradability.png"))


def test_plot_factor_cumulative_overview(tmp_path):
    index = pd.date_range("2010-01-01", periods=24, freq="MS")
    self_built = {
        universe: pd.DataFrame(
            {"SMB": np.linspace(-0.02, 0.03, 24), "WML": np.linspace(-0.01, 0.04, 24)},
            index=index,
        )
        for universe in ("us", "sg")
    }
    reference = pd.DataFrame(
        {"SMB": np.linspace(-0.01, 0.02, 24), "HML": np.linspace(-0.02, 0.01, 24)},
        index=index,
    )
    _assert_png(
        viz.plot_factor_cumulative_overview(
            self_built, reference, tmp_path / "factor_overview.png"
        )
    )


def test_plot_index_concentration(tmp_path):
    index = pd.date_range("2010-01-01", periods=12, freq="MS")

    def panel(k: int) -> pd.DataFrame:
        base = np.linspace(1.0, float(k), k)
        values = np.tile(base / base.sum(), (len(index), 1))
        return pd.DataFrame(values, index=index, columns=[f"a{i}" for i in range(k)])

    _assert_png(
        viz.plot_index_concentration(
            {"us": panel(8), "sg": panel(6)}, tmp_path / "concentration.png"
        )
    )
