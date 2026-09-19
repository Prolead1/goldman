"""Tests for the data loaders and the portfolio-choice exhibits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goldman import config
from goldman.analysis.portfolio_choice import (
    representative_weights,
    run_portfolio_analysis,
    summary_statistics,
)
from goldman.data import (
    load_ff3,
    load_returns,
    monthly_rf,
    prices_to_returns,
    to_period_index,
)

SYNTHETIC = config.universe_path("synthetic")
HAS_SYNTHETIC = Path(SYNTHETIC).exists()


def test_prices_to_returns_drops_fully_missing_months():
    prices = pd.DataFrame(
        {
            "a": [100.0, 110.0, 121.0, 133.1],
            "b": [50.0, np.nan, np.nan, np.nan],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="MS"),
    )
    returns = prices_to_returns(prices, min_obs=2)
    assert list(returns.columns) == ["a"]
    assert len(returns) == 3
    assert returns["a"].iloc[0] == pytest.approx(0.10)


def test_load_ff3_parses_period_index():
    if not config.data_path(config.FF3_FILENAME).exists():
        pytest.skip("ff3_monthly.csv not present")
    ff3 = load_ff3()
    assert isinstance(ff3.index, pd.PeriodIndex)
    assert {"Mkt-RF", "SMB", "HML", "RF"}.issubset(ff3.columns)
    assert ff3["RF"].abs().max() < 0.1  # stored as decimals, not percent


def test_monthly_rf_falls_back_outside_ff3_coverage():
    ff3 = pd.DataFrame(
        {"RF": [0.0002, 0.0003]},
        index=pd.PeriodIndex(["2016-09", "2016-10"], freq="M"),
    )
    index = pd.to_datetime(["2016-09-01", "2030-01-01"])
    rf = monthly_rf(index, ff3=ff3, fallback=0.003)
    assert rf.iloc[0] == pytest.approx(0.0002)
    assert rf.iloc[1] == pytest.approx(0.003)


def test_summary_statistics_columns():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0.01, 0.04, size=(24, 3)),
        columns=list("ABC"),
        index=pd.date_range("2020-01-01", periods=24, freq="MS"),
    )
    stats = summary_statistics(returns, rf=0.003)
    assert list(stats.columns) == [
        "mean (ann.)",
        "vol (ann.)",
        "sharpe (ann.)",
        "min month",
        "max month",
    ]


def test_to_period_index_converts_to_monthly_periods():
    frame = pd.DataFrame(
        {"x": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-01", "2020-02-01"])
    )
    out = to_period_index(frame)
    assert isinstance(out.index, pd.PeriodIndex)
    assert out.index.freqstr == "M"
    assert list(out["x"]) == [1.0, 2.0]


def test_load_ff3_percent_values_are_rescaled(tmp_path):
    path = tmp_path / "ff3.csv"
    pd.DataFrame(
        {"Date": ["2020-01"], "Mkt-RF": [2.0], "SMB": [1.0], "HML": [0.5], "RF": [0.1]}
    ).to_csv(path, index=False)
    ff3 = load_ff3(path=path, as_decimals=False)
    assert ff3.loc[pd.Period("2020-01", "M"), "Mkt-RF"] == pytest.approx(0.02)


def test_monthly_rf_uses_constant_without_factor_file():
    index = pd.to_datetime(["2020-01-01", "2020-02-01"])
    rf = monthly_rf(index, ff3=None, fallback=0.004)
    assert (rf == 0.004).all()
    assert rf.name == "RF"


def test_monthly_rf_falls_back_when_rf_column_missing():
    ff3 = pd.DataFrame({"Mkt-RF": [0.01]}, index=pd.PeriodIndex(["2020-01"], freq="M"))
    index = pd.to_datetime(["2020-01-01"])
    rf = monthly_rf(index, ff3=ff3, fallback=0.002)
    assert rf.iloc[0] == pytest.approx(0.002)


def test_summary_statistics_accepts_rf_series():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        rng.normal(0.01, 0.04, size=(24, 3)),
        columns=list("ABC"),
        index=pd.date_range("2020-01-01", periods=24, freq="MS"),
    )
    rf = pd.Series(0.003, index=returns.index)
    stats = summary_statistics(returns, rf=rf)
    assert stats["sharpe (ann.)"].notna().all()


def test_representative_weights_variants():
    weights = pd.DataFrame(
        {"a": [0.2, 0.4], "b": [0.8, 0.6]},
        index=pd.to_datetime(["2020-01-01", "2020-02-01"]),
    )
    assert representative_weights(weights, "first")["a"] == pytest.approx(0.2)
    assert representative_weights(weights, "last")["a"] == pytest.approx(0.4)
    series = pd.Series({"a": 1.0})
    assert representative_weights(series) is series
    with pytest.raises(ValueError):
        representative_weights(weights, "median")


@pytest.mark.skipif(not HAS_SYNTHETIC, reason="synthetic sample not present")
def test_run_portfolio_analysis_with_value_weight_panel(tmp_path):
    returns = load_returns(universe="synthetic", min_obs=12)
    weights = pd.DataFrame(
        1.0 / returns.shape[1], index=returns.index, columns=returns.columns
    )
    result = run_portfolio_analysis(
        returns, comparison=weights, comparison_label="value weights", output_dir=tmp_path
    )
    assert result.output_paths["value_weights"].exists()


@pytest.mark.skipif(not HAS_SYNTHETIC, reason="synthetic sample not present")
def test_run_portfolio_analysis_writes_all_exhibits(tmp_path):
    returns = load_returns(universe="synthetic", min_obs=12)
    result = run_portfolio_analysis(returns, output_dir=tmp_path)

    for key in ("summary", "weights", "subsample"):
        assert result.output_paths[key].exists()
    assert result.output_paths["weights"].suffix == ".png"
    assert -1.0 <= result.weight_correlation <= 1.0
    assert result.most_extreme_weight > 0.0