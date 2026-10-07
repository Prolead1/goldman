"""Tests for the expanding-window out-of-sample validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.out_of_sample import (
    OOS_BENCHMARKS,
    out_of_sample_summary,
    run_out_of_sample_analysis,
    walk_forward_returns,
)
from goldman.analysis.portfolio import equal_weights
from goldman.analysis.robustness import VARIANTS


def _returns(n: int = 120, k: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2010-01-01", periods=n, freq="MS")
    market = rng.normal(0.006, 0.04, n)
    data = {
        f"a{i}": 0.002 + (0.5 + 0.2 * i) * market + rng.normal(0.0, 0.02, n)
        for i in range(k)
    }
    return pd.DataFrame(data, index=index)


def test_walk_forward_starts_after_the_training_window():
    returns = _returns()
    oos = walk_forward_returns(returns, 0.001, min_train=60)
    assert oos.index[0] == returns.index[60]
    assert oos.index[-1] == returns.index[-1]
    assert len(oos) == len(returns) - 60
    assert set(VARIANTS).issubset(oos.columns)
    assert "equal_weight" in oos.columns
    assert "index" not in oos.columns  # no index supplied


def test_equal_weight_column_is_the_one_over_n_portfolio():
    returns = _returns()
    oos = walk_forward_returns(returns, 0.001, min_train=60)
    expected = (returns @ equal_weights(returns.columns) - 0.001).iloc[60:]
    pd.testing.assert_series_equal(
        oos["equal_weight"], expected.rename("equal_weight"), check_names=False
    )


def test_supplied_index_is_forwarded_on_the_same_dates():
    returns = _returns()
    index_returns = (returns.mean(axis=1) - 0.001).rename("index")
    oos = walk_forward_returns(returns, 0.001, min_train=60, index_returns=index_returns)
    assert "index" in oos.columns
    pd.testing.assert_series_equal(
        oos["index"], index_returns.iloc[60:].rename("index"), check_names=False
    )


def test_min_train_validation():
    returns = _returns()
    with pytest.raises(ValueError):
        walk_forward_returns(returns, 0.001, min_train=1)
    with pytest.raises(ValueError):
        walk_forward_returns(returns, 0.001, min_train=len(returns))


def test_summary_reports_risk_and_return_statistics():
    returns = _returns()
    oos = walk_forward_returns(returns, 0.001, min_train=60)
    summary = out_of_sample_summary(oos)
    assert list(summary.index) == list(oos.columns)
    assert set(summary.columns) == {"months", "ann. mean", "ann. vol", "Sharpe", "max drawdown"}
    assert (summary["ann. vol"] > 0).all()
    assert (summary["max drawdown"] <= 0).all()
    assert summary["months"].eq(len(oos)).all()


def test_run_out_of_sample_analysis_writes_both_exhibits(tmp_path):
    returns = _returns()
    index_returns = (returns.mean(axis=1) - 0.001).rename("index")
    result = run_out_of_sample_analysis(
        returns,
        universe="test",
        rf=0.001,
        index_returns=index_returns,
        min_train=60,
        output_dir=tmp_path,
        label="test",
    )
    for key in ("growth", "sharpe"):
        path = result.output_paths[key]
        assert path.exists()
        assert path.stat().st_size > 1000
    assert "index" in result.returns.columns
    assert result.growth.iloc[-1].notna().all()
    assert set(OOS_BENCHMARKS).issubset(result.returns.columns)
