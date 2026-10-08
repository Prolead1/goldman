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
    align_excess_returns,
    load_developed,
    load_factors,
    load_ff3,
    load_ff5,
    load_holding_value_weights,
    load_index_proxy_returns,
    load_long_history,
    load_momentum,
    load_returns,
    load_test_portfolios,
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

# ----------------------------------------------- Ken French factors & portfolios
def _kf_csv(path, columns, values):
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "Date", ["2020-01", "2020-02", "2020-03"][: len(frame)])
    frame.to_csv(path, index=False)
    return path


def test_load_factors_joins_ff3_and_momentum(tmp_path):
    ff3 = _kf_csv(
        tmp_path / "ff3.csv",
        ["Mkt-RF", "SMB", "HML", "RF"],
        [[0.01, 0.0, 0.0, 0.001], [0.02, 0.0, 0.0, 0.001], [0.03, 0.0, 0.0, 0.001]],
    )
    mom = _kf_csv(tmp_path / "mom.csv", ["Mom"], [[0.005], [0.006]])
    factors = load_factors(ff3_path=ff3, momentum_path=mom)
    assert list(factors.columns) == ["Mkt-RF", "SMB", "HML", "RF", "Mom"]
    assert len(factors) == 2  # inner join on the months both cover
    assert isinstance(factors.index, pd.PeriodIndex)


def test_committed_ken_french_files_load():
    if not config.data_path(config.FF5_FILENAME).exists():
        pytest.skip("Ken French files not present")
    ff5 = load_ff5()
    assert list(ff5.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert load_momentum().columns.tolist() == ["Mom"]
    ff25 = load_test_portfolios("ff25")
    assert ff25.shape[1] == 25
    assert ff25.abs().max().max() < 1.0  # decimals, not percent
    assert load_test_portfolios("mom10").shape[1] == 10


def test_load_test_portfolios_rejects_unknown_name(tmp_path):
    with pytest.raises(KeyError, match="ff25"):
        load_test_portfolios("industry49")
    path = _kf_csv(tmp_path / "p.csv", ["P1", "P2"], [[0.01, 0.02]])
    assert load_test_portfolios("custom", path=path).shape == (1, 2)


def test_align_excess_returns_uses_common_months_and_subtracts_rf():
    returns = pd.DataFrame(
        {"a": [0.02, 0.03, 0.04]},
        index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
    )
    factors = pd.DataFrame(
        {"Mkt-RF": [0.01, 0.01], "RF": [0.001, 0.002]},
        index=pd.PeriodIndex(["2020-02", "2020-03"], freq="M"),
    )
    excess, aligned = align_excess_returns(returns, factors)
    assert list(excess.index.astype(str)) == ["2020-02", "2020-03"]
    assert excess["a"].tolist() == pytest.approx([0.029, 0.038])
    assert aligned.index.equals(excess.index)

    with pytest.raises(ValueError, match="RF"):
        align_excess_returns(returns, factors.drop(columns="RF"))
    later = factors.copy()
    later.index = pd.PeriodIndex(["2030-01", "2030-02"], freq="M")
    with pytest.raises(ValueError, match="no months"):
        align_excess_returns(returns, later)


def test_to_period_index_accepts_period_input():
    frame = pd.DataFrame({"x": [1.0]}, index=pd.PeriodIndex(["2020-01"], freq="M"))
    assert to_period_index(frame).index.equals(frame.index)


# ------------------------------------------------- Stage 4 data (proxy, history)
def test_load_developed_is_usd_with_us_tbill():
    if not config.data_path(config.DEVELOPED_FILENAME).exists():
        pytest.skip("developed-markets file not present")
    dev = load_developed()
    ff3 = load_ff3()
    common = dev.index.intersection(ff3.index)
    assert {"Mkt-RF", "RF"}.issubset(dev.columns)
    assert (dev.loc[common, "RF"] - ff3.loc[common, "RF"]).abs().max() < 2e-4


def test_load_long_history_aligns_returns_caps_and_factors():
    if not config.data_path(config.LONG_HISTORY["returns"]).exists():
        pytest.skip("long-history files not present")
    hist = load_long_history()
    assert hist.returns.shape == hist.caps.shape
    assert hist.returns.index.equals(hist.factors.index)
    assert str(hist.returns.index[0]) == "1963-07"
    assert str(hist.factors_full.index[0]) == "1926-07"
    assert (hist.caps > 0).all().all()


def test_load_holding_value_weights_lags_one_month(tmp_path):
    path = tmp_path / "vw.csv"
    pd.DataFrame(
        {
            "asset": ["a", "b", "a", "b", "a", "b"],
            "date": [d for d in ("2020-01-01", "2020-02-01", "2020-03-01") for _ in range(2)],
            "weight": [0.5, 0.5, 0.75, 0.25, 0.1, 0.9],
        }
    ).to_csv(path, index=False)
    months = pd.to_datetime(["2020-02-01", "2020-03-01"])
    held = load_holding_value_weights(months, path=path)
    assert isinstance(held.index, pd.PeriodIndex)
    assert held.loc[pd.Period("2020-02", "M"), "a"] == pytest.approx(0.5)  # January's weights
    assert held.loc[pd.Period("2020-03", "M"), "a"] == pytest.approx(0.75)  # February's


def test_load_index_proxy_returns_labels_and_returns(tmp_path):
    path = tmp_path / "proxies.csv"
    pd.DataFrame(
        {"Date": ["2020-01-01", "2020-02-01", "2020-03-01"], "SPY": [100.0, 110.0, 99.0],
         "URTH": [50.0, 50.0, 55.0]}
    ).to_csv(path, index=False)
    proxies = load_index_proxy_returns(path)
    assert list(proxies.columns) == list(config.INDEX_PROXIES.values())
    assert proxies.iloc[0, 0] == pytest.approx(0.10)
    assert len(proxies) == 2
