"""Tests for the universe registry, risk-free providers and market proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman import config
from goldman.data import (
    load_rf_file,
    market_excess_returns,
    monthly_rf_from_sora_export,
    risk_free_series,
)


def _dates(n: int = 6) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="MS")


# --------------------------------------------------------------------- registry
def test_registry_covers_both_markets():
    assert {"us", "sg"}.issubset(set(config.universe_names()))
    assert config.get_universe("us").currency == "USD"
    assert config.get_universe("sg").currency == "SGD"


def test_universe_paths_follow_the_registry():
    assert config.universe_path("us").name == config.get_universe("us").prices_file
    assert config.value_weights_path("sg") == config.data_path("value_weights_sg.csv")
    assert config.value_weights_path("synthetic") is None


def test_unknown_universe_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown universe"):
        config.get_universe("mars")
    with pytest.raises(KeyError):
        config.universe_path("mars")


# ------------------------------------------------------------------ risk-free
def test_risk_free_constant_for_constant_source():
    index = _dates()
    rf = risk_free_series("synthetic", index)
    assert (rf == config.RF_MONTHLY).all()
    assert rf.index.equals(index)


def test_risk_free_ff3_source_uses_supplied_factor_frame():
    index = _dates(2)
    ff3 = pd.DataFrame(
        {"RF": [0.001, 0.002]},
        index=pd.PeriodIndex(["2020-01", "2020-02"], freq="M"),
    )
    rf = risk_free_series("us", index, ff3=ff3)
    assert rf.tolist() == pytest.approx([0.001, 0.002])


def test_load_rf_file_parses_date_column(tmp_path):
    path = tmp_path / "rf.csv"
    pd.DataFrame({"Date": ["2020-01", "2020-02"], "RF": [0.001, 0.002]}).to_csv(path, index=False)
    series = load_rf_file(path)
    assert isinstance(series.index, pd.PeriodIndex)
    assert series.loc[pd.Period("2020-02", "M")] == pytest.approx(0.002)


def test_load_rf_file_requires_rf_column(tmp_path):
    path = tmp_path / "rf.csv"
    pd.DataFrame({"Date": ["2020-01"], "rate": [0.001]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="RF"):
        load_rf_file(path)


def test_monthly_rf_from_sora_export_compounds_the_index(tmp_path):
    path = tmp_path / "mas.csv"
    path.write_text(
        "MAS: test export\n"
        "\n"
        "Domestic Interest Rates (Daily)\n"
        "\n"
        "\n"
        "SORA Value Date,,,SORA Publication Date,SORA,SORA Index,Compound SORA - 1 month\n"
        "2019,Dec,31,01 Jan 2020,1.50,1.0000000000,1.50\n"
        "2020,Jan,31,01 Feb 2020,1.50,1.0100000000,1.50\n"
        "2020,Feb,29,02 Mar 2020,1.50,1.0200000000,1.50\n"
    )
    out = monthly_rf_from_sora_export(path)
    # December is the NaN base and is dropped; later months are index ratios.
    assert out["Date"].tolist() == ["2020-01", "2020-02"]
    assert out["RF"].iloc[0] == pytest.approx(0.01)
    assert out["RF"].iloc[1] == pytest.approx(1.02 / 1.01 - 1.0)


def test_monthly_rf_from_sora_export_rejects_other_csvs(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("Date,RF\n2020-01,0.001\n")
    with pytest.raises(ValueError, match="MAS SORA export"):
        monthly_rf_from_sora_export(path)


def test_risk_free_file_source_aligned_and_filled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    pd.DataFrame({"Date": ["2020-01"], "RF": [0.004]}).to_csv(
        tmp_path / "rf_sg_monthly.csv", index=False
    )
    index = _dates(3)
    rf = risk_free_series("sg", index)
    # January is the stated rate; later months carry it forward.
    assert rf.tolist() == pytest.approx([0.004, 0.004, 0.004])


def test_risk_free_file_source_warns_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.warns(UserWarning, match="falling back to a constant"):
        rf = risk_free_series("sg", _dates())
    assert (rf == config.get_universe("sg").rf_constant).all()


# --------------------------------------------------------------- market proxy
def _panel(n: int = 4):
    dates = _dates(n)
    returns = pd.DataFrame(
        {"a": [0.01, 0.02, -0.01, 0.03], "b": [0.00, 0.01, 0.02, -0.02]}, index=dates
    )
    weights = pd.DataFrame({"a": [0.5] * n, "b": [0.5] * n}, index=dates)
    return returns, weights


def test_value_weighted_market_excess_matches_manual():
    returns, weights = _panel()
    market = market_excess_returns(returns, value_weights=weights, rf=0.001)
    expected = (weights * returns).sum(axis=1) - 0.001
    np.testing.assert_allclose(market.to_numpy(), expected.to_numpy())
    assert market.name == "market_excess"


def test_value_weighted_needs_universe_or_weights():
    returns, _ = _panel()
    with pytest.raises(ValueError, match="universe"):
        market_excess_returns(returns, universe=None, value_weights=None)


def test_ff_market_uses_factor_column():
    returns, _ = _panel()
    ff3 = pd.DataFrame(
        {"Mkt-RF": [0.01, 0.02, 0.03, 0.04]},
        index=pd.PeriodIndex(["2020-01", "2020-02", "2020-03", "2020-04"], freq="M"),
    )
    market = market_excess_returns(returns, method="ff_market", ff3=ff3)
    assert market.tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04])


def test_ff_market_rejects_incomplete_calendar():
    returns, _ = _panel()
    ff3 = pd.DataFrame(
        {"Mkt-RF": [0.01]},
        index=pd.PeriodIndex(["2020-01"], freq="M"),
    )
    with pytest.raises(ValueError, match="cover the whole"):
        market_excess_returns(returns, method="ff_market", ff3=ff3)


def test_unknown_market_proxy_raises():
    returns, weights = _panel()
    with pytest.raises(ValueError, match="Unknown market proxy"):
        market_excess_returns(returns, method="magic", value_weights=weights, rf=0.0)
