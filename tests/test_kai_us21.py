"""Checks for timing and return accounting in the individual comparison."""

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.kai_us21 import estimate, inputs, lag_weights, metrics, portfolio_returns, run


def test_lag_uses_previous_calendar_month_and_rejects_gaps():
    dates = pd.period_range("2020-01", periods=3, freq="M")
    snapshots = pd.DataFrame({"A": [.8, .2, .6], "B": [.2, .8, .4]}, index=dates)
    result = lag_weights(snapshots, dates[1:])
    assert result.A.tolist() == [.8, .2]
    with pytest.raises(ValueError, match="previous-month"):
        lag_weights(snapshots.iloc[1:], dates[1:])


def test_return_accounting_handles_short_positions_and_column_order():
    returns = pd.DataFrame({"A": [.1, -.2], "B": [-.1, .1]})
    weights = pd.Series({"B": -.5, "A": 1.5})
    np.testing.assert_allclose(portfolio_returns(returns, weights), [.2, -.35])


def test_drawdown_includes_initial_capital():
    returns = pd.DataFrame({"portfolio": [-.2, .1]})
    result = metrics(returns, pd.Series(0., index=returns.index))
    assert result.loc["portfolio", "max_drawdown"] == pytest.approx(-.2)
    assert result.loc["portfolio", "cagr"] == pytest.approx(.88**6 - 1)


def test_pack_uses_training_only_weights_and_consistent_return_accounting(tmp_path):
    out = run(tmp_path)
    returns, factors, _, _ = inputs()
    expected = estimate(returns.iloc[:96], factors.RF.iloc[:96])
    saved = pd.read_csv(out / "holdout_weights.csv", index_col=0).iloc[:, 0]
    np.testing.assert_allclose(saved, expected, rtol=1e-10)
    full = pd.read_csv(out / "portfolio_returns_in_sample.csv", index_col=0)
    holdout = pd.read_csv(out / "portfolio_returns_holdout.csv", index_col=0)
    assert len(holdout) == 22
    np.testing.assert_allclose(holdout.iloc[:, 0], returns.iloc[96:] @ expected)
    np.testing.assert_allclose(full["FF US market"], factors["Mkt-RF"] + factors.RF)
    assert (out / "meeting_report.md").is_file()


def test_cached_selection_has_complete_observed_rf_and_lagged_weights():
    returns, factors, _, beginning = inputs()
    assert returns.shape == (118, 21)
    assert returns.index[0] == pd.Period("2016-10", freq="M")
    assert returns.index[-1] == pd.Period("2026-07", freq="M")
    assert not factors.isna().any().any()
    assert not beginning.isna().any().any()
    np.testing.assert_allclose(beginning.sum(axis=1), 1.)
