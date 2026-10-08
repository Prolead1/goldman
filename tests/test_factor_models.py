"""Tests for the Stage 3 factor-model exhibits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.factor_models import (
    MODEL_FACTORS,
    alpha_comparison,
    anomaly_verdict,
    factor_loadings,
    factor_summary,
    fit_models,
    model_summary,
    plot_alpha_comparison,
    run_factor_analysis,
    standard_models,
)


def _factor_world(seed: int = 0, n_obs: int = 600):
    """Four factors with positive premia and four assets with known stories.

    * ``size_tilt`` — no true alpha, but a big SMB loading: a CAPM "anomaly"
      that FF3 absorbs.
    * ``true_alpha`` — a genuine alpha no model explains.
    * ``market_only`` — the CAPM is right.
    * ``hidden`` — a true alpha masked under the CAPM by a negative SMB
      loading; it emerges once SMB is controlled for.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1970-01", periods=n_obs, freq="M")
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.006, 0.04, n_obs),
            "SMB": rng.normal(0.008, 0.02, n_obs),
            "HML": rng.normal(0.003, 0.02, n_obs),
            "Mom": rng.normal(0.005, 0.03, n_obs),
            "RF": np.full(n_obs, 0.002),
        },
        index=idx,
    )
    noise = rng.normal(0.0, 0.004, (n_obs, 4))
    mkt, smb = factors["Mkt-RF"], factors["SMB"]
    excess = pd.DataFrame(
        {
            "size_tilt": mkt + 1.5 * smb + noise[:, 0],
            "true_alpha": 0.004 + mkt + noise[:, 1],
            "market_only": mkt + noise[:, 2],
            "hidden": 0.008 + mkt - 1.0 * smb + noise[:, 3],
        },
        index=idx,
    )
    return excess, factors


def _fits():
    excess, factors = _factor_world()
    return excess, factors, fit_models(excess, standard_models(factors, excess.index))


def test_standard_models_builds_nested_sequence():
    excess, factors = _factor_world()
    models = standard_models(factors, excess.index)
    assert list(models) == list(MODEL_FACTORS)
    assert list(models["Carhart4"].columns) == ["Mkt-RF", "SMB", "HML", "Mom"]

    ff5 = factors.assign(RMW=0.001, CMA=0.001)
    with_ff5 = standard_models(factors, excess.index, ff5=ff5)
    assert list(with_ff5["FF5"].columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    with pytest.raises(ValueError, match="cover"):
        standard_models(factors.iloc[1:], excess.index)


def test_factor_summary_reports_premia():
    _, factors = _factor_world()
    summary = factor_summary(factors[["Mkt-RF", "SMB"]])
    assert list(summary.columns) == [
        "mean (ann.)",
        "t(mean)",
        "vol (ann.)",
        "sharpe (ann.)",
        "months",
    ]
    smb = factors["SMB"]
    assert summary.loc["SMB", "t(mean)"] == pytest.approx(
        smb.mean() / (smb.std() / np.sqrt(len(smb)))
    )


def test_anomaly_verdict_classifies_each_story():
    _, _, fits = _fits()
    verdict = anomaly_verdict(fits)["verdict"]
    assert verdict["size_tilt"] == "absorbed by FF3"
    assert verdict["true_alpha"] == "survives all models"
    assert verdict["market_only"] == "no anomaly"
    assert verdict["hidden"] == "emerges under FF3"
    with pytest.raises(KeyError):
        anomaly_verdict(fits, base="APT")


def test_alpha_comparison_and_model_summary_shapes():
    excess, _, fits = _fits()
    comparison = alpha_comparison(fits)
    assert comparison.shape == (excess.shape[1], 3 * len(fits))
    assert "FF3 t(alpha)" in comparison.columns

    summary = model_summary(fits)
    assert list(summary.index) == list(MODEL_FACTORS)
    assert summary.loc["FF3", "avg R2"] > summary.loc["CAPM", "avg R2"]
    assert summary.loc["CAPM", "factors"] == 1


def test_factor_loadings_layout():
    _, _, fits = _fits()
    loadings = factor_loadings(fits["FF3"])
    assert list(loadings.columns) == [
        "Mkt-RF loading",
        "Mkt-RF t",
        "SMB loading",
        "SMB t",
        "HML loading",
        "HML t",
        "R2",
    ]
    assert loadings.loc["size_tilt", "SMB loading"] == pytest.approx(1.5, abs=0.05)


def test_plot_alpha_comparison_limits_models(tmp_path):
    _, _, fits = _fits()
    path = plot_alpha_comparison(fits, tmp_path / "alphas.png")
    assert path.exists()
    too_many = {f"m{i}": fits["CAPM"] for i in range(5)}
    with pytest.raises(ValueError, match="At most"):
        plot_alpha_comparison(too_many, tmp_path / "x.png")


def test_run_factor_analysis_writes_every_exhibit(tmp_path):
    excess, factors = _factor_world()
    models = standard_models(factors, excess.index)
    result = run_factor_analysis(excess, models, output_dir=tmp_path, prefix="t_")
    assert set(result.output_paths) == {"comparison", "summary", "verdict", "loadings", "chart"}
    for path in result.output_paths.values():
        assert path.exists() and path.name.startswith("t_")
    assert "Mom loading" in result.loadings.columns  # richest model's loadings
