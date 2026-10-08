"""Tests for the self-built momentum factor and the appendix robustness helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.capm import alpha_joint_test, capm_regressions
from goldman.analysis.portfolio import two_fund_allocation
from goldman.analysis.portfolio_choice import subset_tangency_check, summary_statistics
from goldman.analysis.self_built import factor_report, long_short_factor, momentum_signal


def test_momentum_signal_skips_the_last_month():
    r = pd.DataFrame({"a": np.arange(1, 16) / 100.0})
    sig = momentum_signal(r, lookback=12, skip=1)
    assert sig.iloc[:12].isna().all().all()
    # Month 12 uses months 0..10 (t-12 .. t-2), not month 11 (t-1).
    assert sig.iloc[12, 0] == pytest.approx(np.prod(1 + r.iloc[0:11, 0]) - 1)
    assert sig.iloc[13, 0] == pytest.approx(np.prod(1 + r.iloc[1:12, 0]) - 1)


def test_long_short_factor_picks_extremes_and_counts_turnover():
    idx = pd.period_range("2000-01", periods=3, freq="M")
    signal = pd.DataFrame(
        {"a": [3.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0], "c": [1.0, 3.0, 3.0], "d": [np.nan] * 3},
        index=idx,
    )
    returns = pd.DataFrame(
        {"a": [0.1, 0.0, 0.0], "b": 0.0, "c": [-0.1, 0.2, 0.0], "d": 0.0}, index=idx
    )
    built = long_short_factor(returns, signal, n_side=1, name="mom")
    assert built.weights.loc[idx[0], "a"] == 1.0 and built.weights.loc[idx[0], "c"] == -1.0
    assert built.returns.iloc[0] == pytest.approx(0.2)  # long a (+10%), short c (-10%)
    assert built.returns.iloc[1] == pytest.approx(0.2)  # now long c, short a
    assert np.isnan(built.turnover.iloc[0])
    assert built.turnover.iloc[1] == pytest.approx(4.0)  # both legs flip
    assert built.turnover.iloc[2] == pytest.approx(0.0)
    assert (built.weights["d"] == 0).all()


def test_long_short_factor_skips_thin_months():
    idx = pd.period_range("2000-01", periods=2, freq="M")
    signal = pd.DataFrame({"a": [1.0, 1.0], "b": [np.nan, 2.0]}, index=idx)
    returns = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]}, index=idx)
    built = long_short_factor(returns, signal, n_side=1)
    assert list(built.returns.index) == [idx[1]]


def test_factor_report_recovers_a_known_premium():
    rng = np.random.default_rng(0)
    idx = pd.period_range("1970-01", periods=600, freq="M")
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.005, 0.04, 600),
            "SMB": rng.normal(0.0, 0.02, 600),
            "HML": rng.normal(0.0, 0.02, 600),
        },
        index=idx,
    )
    ret = pd.Series(0.01 + rng.normal(0, 0.02, 600), index=idx, name="f")
    built_turnover = pd.Series(0.5, index=idx)
    from goldman.analysis.self_built import SelfBuiltFactor

    built = SelfBuiltFactor(returns=ret, weights=pd.DataFrame(index=idx), turnover=built_turnover)
    rep = factor_report(built, factors, reference=ret * 2)
    assert rep["months"] == 600
    assert rep["CAPM alpha (ann.)"] == pytest.approx(0.12, abs=0.03)
    assert rep["t(CAPM alpha)"] > 5
    assert rep["corr with Ken French Mom"] == pytest.approx(1.0)
    assert rep["mean net of 50bp (ann.)"] == pytest.approx(rep["mean (ann.)"] - 0.03)
    assert rep["break-even cost (bp)"] == pytest.approx(ret.mean() / 0.5 * 1e4)


def test_two_fund_allocation_formula():
    market = pd.Series([0.02, -0.01, 0.015, 0.005] * 30)
    table = two_fund_allocation(market, risk_aversions=(2.0, 4.0))
    mean, var = market.mean() * 12, market.var() * 12
    assert table.loc[2.0, "risky share y*"] == pytest.approx(mean / (2 * var))
    assert table.loc[4.0, "risky share y*"] == pytest.approx(table.loc[2.0, "risky share y*"] / 2)
    assert table.loc[2.0, "y* if premium - 2se"] < table.loc[2.0, "risky share y*"]
    assert table.attrs["months"] == 120


def test_subset_tangency_check_reports_conditioning():
    rng = np.random.default_rng(1)
    base = rng.normal(0.01, 0.04, (120, 4))
    near_copy = base[:, :2].mean(axis=1, keepdims=True) + rng.normal(0, 0.002, (120, 1))
    excess = pd.DataFrame(np.hstack([base, near_copy]), columns=list("abcde"))
    table = subset_tangency_check(excess, {"all": list("abcde"), "no copy": list("abcd")})
    cond = table["condition number (corr)"]
    assert cond["all"] > cond["no copy"]
    assert table.loc["no copy", "assets"] == 4


def test_summary_sharpe_uses_excess_return_volatility():
    idx = pd.date_range("2020-01-01", periods=4, freq="MS")
    returns = pd.DataFrame({"a": [0.02, 0.01, 0.03, 0.00]}, index=idx)
    rf = pd.Series([0.001, 0.002, 0.003, 0.004], index=idx)
    sharpe = summary_statistics(returns, rf=rf).loc["a", "sharpe (ann.)"]
    ex = returns["a"] - rf
    assert sharpe == pytest.approx(round(ex.mean() / ex.std() * np.sqrt(12), 3))


def test_joint_test_reports_minimum_detectable_alpha():
    rng = np.random.default_rng(2)
    idx = pd.period_range("2000-01", periods=120, freq="M")
    market = pd.Series(rng.normal(0.006, 0.04, 120), index=idx, name="Mkt-RF")
    excess = pd.DataFrame(
        np.outer(market, [0.8, 1.0, 1.2]) + rng.normal(0, 0.03, (120, 3)), index=idx
    )
    fit = capm_regressions(excess, market)
    joint = alpha_joint_test(fit)
    se = (fit.alpha / fit.tstats["alpha"]).abs().median() * 12
    assert joint["min detectable alpha (ann., median asset)"] == pytest.approx(2.8016 * se)
