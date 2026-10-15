"""Tests for the custom factors: 52-week high and volatility-managed momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman import viz
from goldman.analysis import custom_factors as cf
from goldman.analysis.factor_construction import (
    long_short_spread,
    long_short_weights,
    momentum_factor,
)


def _panel(seed: int = 0, n_obs: int = 60, n_assets: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-01", periods=n_obs, freq="MS")
    data = rng.normal(0.008, 0.06, (n_obs, n_assets))
    return pd.DataFrame(data, index=index, columns=[f"a{i}" for i in range(n_assets)])


def test_long_short_weights_reproduce_the_spread():
    returns = _panel()
    signal = returns.shift(1)
    weights = long_short_weights(returns, signal, n_quantiles=3)
    spread = long_short_spread(returns, signal, n_quantiles=3)
    rebuilt = (weights * returns).sum(axis=1).where(weights.abs().sum(axis=1) > 0)
    pd.testing.assert_series_equal(rebuilt, spread, check_names=False, check_freq=False)
    active = weights.abs().sum(axis=1) > 0
    np.testing.assert_allclose(weights[active].sum(axis=1), 0.0, atol=1e-12)  # dollar neutral
    with pytest.raises(ValueError):
        long_short_weights(returns, signal, n_quantiles=1)


def test_near_high_signal_uses_only_past_prices():
    returns = pd.DataFrame({"a": [0.10, -0.50, 0.20, 0.0, 0.0]},
                           index=pd.date_range("2020-01-01", periods=5, freq="MS"))
    signal = cf.near_high_signal(returns, window=2)
    level = (1 + returns["a"]).cumprod()
    # Month 3 (index 3) sees levels of months 1 and 2 only.
    assert signal.iloc[3, 0] == pytest.approx(level.iloc[2] / max(level.iloc[1], level.iloc[2]))
    assert signal.iloc[:2, 0].isna().all()
    assert (signal.dropna() <= 1.0 + 1e-12).all().all()
    with pytest.raises(ValueError):
        cf.near_high_signal(returns, window=1)


def test_near_high_weights_long_the_asset_at_its_high():
    index = pd.date_range("2020-01-01", periods=14, freq="MS")
    returns = pd.DataFrame(
        {"up": 0.02, "choppy": [0.03, -0.05] * 7, "down": -0.02}, index=index
    )
    weights = cf.near_high_weights(returns, window=12, n_quantiles=3)
    last = weights.iloc[-1]
    assert last["up"] > 0 and last["down"] < 0 and last["choppy"] == 0


def test_volatility_scale_targets_the_volatility():
    factor = pd.Series([0.02, -0.02] * 10, index=pd.date_range("2020-01-01", periods=20, freq="MS"))
    scale = cf.volatility_scale(factor, window=6, target_vol=0.12)
    sigma = factor.iloc[:6].std() * np.sqrt(12)
    assert scale.iloc[6] == pytest.approx(0.12 / sigma)
    assert scale.iloc[:6].isna().all()
    with pytest.raises(ValueError):
        cf.volatility_scale(factor, window=1)


def test_turnover_counts_formation_and_flips():
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    weights = pd.DataFrame(
        {"a": [0.0, 1.0, -1.0, -1.0], "b": [0.0, -1.0, 1.0, 1.0]}, index=index
    )
    traded = cf.turnover(weights)
    assert np.isnan(traded.iloc[0])
    assert traded.tolist()[1:] == [2.0, 4.0, 0.0]


def test_max_drawdown():
    returns = pd.Series([0.10, -0.50, 0.20])
    assert cf.max_drawdown(returns) == pytest.approx(-0.5)
    assert cf.max_drawdown(pd.Series([-0.2, 0.1])) == pytest.approx(-0.2)


def test_build_custom_factors_matches_wml_and_scales_it():
    returns = _panel(1, n_obs=72)
    factors, turnovers = cf.build_custom_factors(returns)
    pd.testing.assert_series_equal(
        factors["WML"].dropna(),
        momentum_factor(returns).dropna(),
        check_names=False,
        check_freq=False,
    )
    scale = cf.volatility_scale(factors["WML"])
    np.testing.assert_allclose(
        factors["VWML"].dropna(), (factors["WML"] * scale).dropna(), rtol=1e-12
    )
    assert list(turnovers.columns) == ["WML", "NEAR52", "VWML"]
    assert (turnovers.dropna() >= 0).all().all()


def test_scorecard_and_spanning():
    returns = _panel(2, n_obs=96)
    factors, turnovers = cf.build_custom_factors(returns)
    market = returns.mean(axis=1).rename("Mkt-RF")
    card = cf.factor_scorecard(factors, market, turnovers=turnovers, cost_bps=50)
    assert set(card.index) == {"WML", "NEAR52", "VWML"}
    row = card.loc["WML"]
    series = factors["WML"].loc[factors.dropna().index]
    assert row["ann. mean"] == pytest.approx(series.mean() * 12)
    assert row["Sharpe"] == pytest.approx(series.mean() / series.std() * np.sqrt(12))
    break_even = series.mean() / row["turnover / month"] * 1e4
    assert row["break-even cost (bp)"] == pytest.approx(break_even)

    # A factor that is exactly 0.5 x a control plus a constant has that constant as alpha.
    control = factors["WML"].dropna()
    copy = 0.5 * control + 0.001
    span = cf.spanning_regression({"copy": (copy, control.to_frame("WML"))})
    assert span.loc["copy", "alpha (ann.)"] == pytest.approx(0.012)
    assert span.loc["copy", "beta WML"] == pytest.approx(0.5)


def test_custom_factor_plots(tmp_path):
    returns = _panel(3, n_obs=84)
    factors, turnovers = cf.build_custom_factors(returns)
    market = returns.mean(axis=1).rename("Mkt-RF")
    panel = pd.concat([factors, market], axis=1)
    panels = {"us": panel, "sg": panel}
    cards = {u: cf.factor_scorecard(factors, market, turnovers=turnovers) for u in ("us", "sg")}
    assert viz.plot_custom_factor_growth_paired(panels, tmp_path / "g.png").exists()
    assert viz.plot_custom_factor_scorecard(cards, tmp_path / "c.png").exists()
