"""Tests for the Stage 4 out-of-sample horse race."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.backtest import (
    STANDARD_RULES,
    Window,
    historical_premium,
    holding_weights,
    performance_table,
    plot_cumulative,
    rolling_backtest,
    sharpe_difference_test,
    tangency_shrinkage,
    tangency_sign_flips,
    value_weighted,
)


def _panel(seed: int = 0, n_obs: int = 90, n_assets: int = 6):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n_obs, freq="M")
    m = rng.normal(0.006, 0.04, (n_obs, 1))
    y = m * rng.uniform(0.6, 1.4, n_assets) + rng.normal(0.001, 0.03, (n_obs, n_assets))
    cols = [f"a{i}" for i in range(n_assets)]
    excess = pd.DataFrame(y, index=idx, columns=cols)
    caps = pd.DataFrame(rng.uniform(1, 5, (n_obs, n_assets)), index=idx, columns=cols)
    held = caps.div(caps.sum(axis=1), axis=0)
    return excess, held


def test_holding_weights_use_previous_month_end():
    idx = pd.period_range("2020-01", periods=3, freq="M")
    month_end = pd.DataFrame({"a": [1.0, 3.0, 1.0], "b": [1.0, 1.0, 3.0]}, index=idx)
    held = holding_weights(month_end)
    assert list(held.index.astype(str)) == ["2020-02", "2020-03"]
    assert held.loc[pd.Period("2020-03", "M"), "a"] == pytest.approx(0.75)  # Feb's caps


def test_value_weighted_rule_earns_lagged_weight_times_return():
    excess, held = _panel()
    res = rolling_backtest(excess, {"vw": value_weighted}, window=24, held_market_weights=held)
    t = excess.index[30]
    expected = float((held.loc[t] * excess.loc[t]).sum())
    assert res.returns.loc[t, "vw"] == pytest.approx(expected)
    assert res.returns.index[0] == excess.index[24]
    assert len(res.returns) == len(excess) - 24


def test_rules_never_see_the_month_they_trade():
    excess, held = _panel(1)
    seen: list[pd.Period] = []

    def spy(win: Window) -> pd.Series:
        seen.append(win.excess.index[-1])
        return pd.Series(1.0 / excess.shape[1], index=excess.columns)

    res = rolling_backtest(excess, {"spy": spy}, window=24, held_market_weights=held)
    for last_seen, traded in zip(seen, res.returns.index, strict=True):
        assert last_seen < traded


def test_standard_rules_run_and_are_fully_invested():
    excess, held = _panel(2)
    prem = pd.Series(0.005, index=excess.index)
    res = rolling_backtest(
        excess, STANDARD_RULES, window=36, held_market_weights=held,
        rf=pd.Series(0.001, index=excess.index), prior_premium=prem,
    )
    for name, w in res.weights.items():
        np.testing.assert_allclose(w.sum(axis=1).abs(), 1.0, atol=1e-6, err_msg=name)
    assert (res.weights["tangency (long-only)"] >= -1e-9).all().all()
    assert res.turnover.iloc[0].isna().all()
    assert (res.turnover["value weighted"].iloc[1:] >= 0).all()


def test_full_shrinkage_recovers_the_market_portfolio():
    excess, held = _panel(3, n_obs=60)
    win = Window(excess=excess, market_weights=held.iloc[-1], prior_premium=0.005)
    # Means equal to the CAPM prior -> intensity 1 -> tangency = market weights.
    from goldman.analysis.estimators import capm_prior_means, ledoit_wolf_covariance

    cov, _ = ledoit_wolf_covariance(excess)
    prior = capm_prior_means(cov, held.iloc[-1], 0.005)
    shifted = excess - excess.mean() + prior
    w = tangency_shrinkage(Window(shifted, held.iloc[-1], prior_premium=0.005))
    np.testing.assert_allclose(w, held.iloc[-1], atol=1e-8)
    assert tangency_shrinkage(win).sum() == pytest.approx(1.0)


def test_rules_need_their_inputs():
    excess, _ = _panel(4)
    with pytest.raises(ValueError):
        value_weighted(Window(excess))
    with pytest.raises(ValueError):
        tangency_shrinkage(Window(excess, market_weights=pd.Series(1.0, index=excess.columns)))
    with pytest.raises(ValueError, match="window"):
        rolling_backtest(excess, {"vw": value_weighted}, window=len(excess))
    with pytest.raises(ValueError, match="index"):
        rolling_backtest(excess, {"vw": value_weighted}, 24, held_market_weights=excess.iloc[1:])


def test_historical_premium_has_no_look_ahead():
    idx = pd.period_range("2000-01", periods=4, freq="M")
    market = pd.Series([0.01, 0.03, -0.02, 0.10], index=idx)
    prem = historical_premium(market, idx[2:])
    assert prem.iloc[0] == pytest.approx(0.02)  # mean of Jan, Feb only
    assert prem.iloc[1] == pytest.approx((0.01 + 0.03 - 0.02) / 3)  # Mar excluded from Mar


def test_sharpe_difference_test_is_symmetric_and_zero_for_identical():
    rng = np.random.default_rng(5)
    a = pd.Series(rng.normal(0.01, 0.04, 200))
    b = pd.Series(rng.normal(0.005, 0.04, 200))
    z_ab, p_ab = sharpe_difference_test(a, b)
    z_ba, p_ba = sharpe_difference_test(b, a)
    assert z_ab == pytest.approx(-z_ba)
    assert p_ab == pytest.approx(p_ba)
    assert sharpe_difference_test(a, a) == (0.0, 1.0)


def test_performance_table_and_chart(tmp_path):
    excess, held = _panel(6)
    res = rolling_backtest(
        excess, {"vw": value_weighted, "1/N": STANDARD_RULES["1/N"]}, 24,
        held_market_weights=held,
    )
    table = performance_table(res, benchmark="vw", references={"index": excess.mean(axis=1)})
    assert list(table.index) == ["vw", "1/N", "index"]
    assert table.loc["vw", "p vs vw"] == 1.0
    assert table.loc["index", "turnover (monthly)"] == 0.0
    assert -1.0 <= table.loc["vw", "max drawdown"] <= 0.0
    path = plot_cumulative(res.returns, tmp_path / "wealth.png", exclude=["1/N"], dashed=["vw"])
    assert path.exists()
    with pytest.raises(ValueError):
        plot_cumulative(pd.concat([res.returns] * 5, axis=1), tmp_path / "x.png")


def test_max_drawdown_marks_wipeout():
    from goldman.analysis.backtest import _max_drawdown

    assert _max_drawdown(pd.Series([0.1, -1.2, 0.5])) == -1.0
    assert _max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_tangency_sign_flips_reports_every_window():
    excess, _ = _panel(7, n_obs=40)
    flips = tangency_sign_flips(excess, window=24)
    assert len(flips) == 40 - 24 + 1
    assert (flips["gross leverage"] >= 1.0 - 1e-9).all()
    assert flips.index[-1] == excess.index[-1]


def test_performance_table_certainty_equivalents():
    excess, held = _panel(8)
    res = rolling_backtest(excess, {"vw": value_weighted}, 24, held_market_weights=held)
    table = performance_table(res, benchmark="vw", risk_aversions=(1.0, 5.0))
    r = res.returns["vw"]
    assert table.loc["vw", "CEQ g=1 (ann.)"] == pytest.approx(12 * (r.mean() - 0.5 * r.var()))
    assert table.loc["vw", "CEQ g=5 (ann.)"] < table.loc["vw", "CEQ g=1 (ann.)"]
