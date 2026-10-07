"""Smoke tests for the visualisation module: every figure must render."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from goldman import viz
from goldman.analysis import (
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
