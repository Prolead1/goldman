"""Tests for the point-in-time value-weight loaders."""

from __future__ import annotations

import pandas as pd
import pytest

from goldman import config
from goldman.data import market_caps as mc


def test_value_weights_from_market_caps_normalises_series():
    caps = pd.Series({"a": 100.0, "b": 300.0})
    weights = mc.value_weights_from_market_caps(caps)
    assert weights["a"] == pytest.approx(0.25)
    assert weights.sum() == pytest.approx(1.0)


def test_value_weights_from_market_caps_normalises_each_row():
    caps = pd.DataFrame({"a": [100.0, 50.0], "b": [300.0, 50.0]})
    weights = mc.value_weights_from_market_caps(caps)
    assert weights.loc[0, "a"] == pytest.approx(0.25)
    assert weights.loc[1, "a"] == pytest.approx(0.5)
    assert weights.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_value_weights_reject_missing_or_nonpositive_total():
    with pytest.raises(ValueError):
        mc.value_weights_from_market_caps(pd.Series({"a": 1.0, "b": float("nan")}))
    with pytest.raises(ValueError):
        mc.value_weights_from_market_caps(pd.Series({"a": 0.0, "b": 0.0}))


def test_unadjust_close_undoes_later_splits():
    index = pd.to_datetime(["2020-06-01", "2020-08-01", "2020-10-01"])
    close = pd.Series([10.0, 40.0, 40.0], index=index)  # Yahoo split-adjusted
    splits = pd.Series([0.0, 4.0, 0.0], index=index)
    raw = mc.unadjust_close(close, splits)
    # June (pre-split) is scaled back up; the split month and later are not.
    assert raw.loc["2020-06-01"] == pytest.approx(40.0)
    assert raw.loc["2020-08-01"] == pytest.approx(40.0)
    assert raw.loc["2020-10-01"] == pytest.approx(40.0)


def test_market_caps_from_shares_forward_fills_without_lookahead():
    prices = pd.DataFrame(
        {"a": [10.0, 20.0, 30.0]},
        index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
    )
    # Share count reported Dec 2019 covers January and February; the
    # mid-February report is first used in March (never before it lands).
    shares = pd.Series(
        [1_000.0, 2_000.0],
        index=pd.to_datetime(["2019-12-15", "2020-02-15"]),
    ).to_frame("a")
    caps = mc.market_caps_from_shares(prices, shares)
    assert caps.loc["2020-01-01", "a"] == pytest.approx(10.0 * 1_000)
    assert caps.loc["2020-02-01", "a"] == pytest.approx(20.0 * 1_000)
    assert caps.loc["2020-03-01", "a"] == pytest.approx(30.0 * 2_000)


def test_load_value_weights_static_broadcasts_to_dates(tmp_path):
    path = tmp_path / "value_weights.csv"
    pd.DataFrame({"asset": ["a", "b"], "market_cap": [100.0, 100.0]}).to_csv(path, index=False)
    dates = pd.to_datetime(["2020-01-01", "2020-02-01"])

    weights = mc.load_value_weights(path=path, dates=dates, assets=["b", "a"])
    assert list(weights.columns) == ["b", "a"]
    assert list(weights.index) == list(dates)
    assert weights.loc[dates[0], "a"] == pytest.approx(0.5)
    assert weights.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_load_value_weights_static_percent_column(tmp_path):
    path = tmp_path / "value_weights.csv"
    pd.DataFrame({"asset": ["a", "b"], "weight": [25.0, 75.0]}).to_csv(path, index=False)
    weights = mc.load_value_weights(path=path, as_percent=True)
    assert weights.iloc[0]["a"] == pytest.approx(0.25)


def test_load_value_weights_dated_panel_aligns_and_forward_fills(tmp_path):
    path = tmp_path / "value_weights.csv"
    pd.DataFrame(
        {
            "asset": ["a", "a", "b", "b"],
            "date": ["2020-01-01", "2020-03-01", "2020-01-01", "2020-03-01"],
            "market_cap": [100.0, 200.0, 300.0, 200.0],
        }
    ).to_csv(path, index=False)
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])

    weights = mc.load_value_weights(path=path, dates=dates, assets=["a", "b"])
    assert weights.loc["2020-01-01", "a"] == pytest.approx(0.25)
    # February has no snapshot of its own: it carries January's.
    assert weights.loc["2020-02-01", "a"] == pytest.approx(0.25)
    assert weights.loc["2020-03-01", "a"] == pytest.approx(0.5)


def test_load_value_weights_dated_panel_missing_asset_raises(tmp_path):
    path = tmp_path / "value_weights.csv"
    pd.DataFrame(
        {"asset": ["a"], "date": ["2020-01-01"], "market_cap": [1.0]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        mc.load_value_weights(path=path, assets=["a", "b"])


def test_build_value_weights_from_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    raw_dir = tmp_path / "us"
    raw_dir.mkdir(parents=True)
    index = pd.to_datetime(["2020-01-01", "2020-02-01"])
    pd.DataFrame({"close": [10.0, 20.0], "shares": [100.0, 100.0]}, index=index).to_csv(
        raw_dir / "a.csv"
    )
    pd.DataFrame({"close": [5.0, 5.0], "shares": [200.0, 400.0]}, index=index).to_csv(
        raw_dir / "b.csv"
    )

    long = mc.build_value_weights_from_raw("us", assets=["a", "b"])
    assert set(long.columns) == {"asset", "date", "market_cap", "weight"}
    first = long[long["date"] == index[0]]
    assert long["date"].nunique() == 2
    assert first.set_index("asset")["weight"].sum() == pytest.approx(1.0)
    # Feb: a cap = 20*100 = 2000, b cap = 5*400 = 2000 -> equal weights.
    feb = long[long["date"] == index[1]].set_index("asset")["weight"]
    assert feb["a"] == pytest.approx(0.5)


def test_load_value_weights_requires_path_or_universe():
    with pytest.raises(ValueError):
        mc.load_value_weights()
    with pytest.raises(ValueError):
        mc.load_value_weights(path="x.csv", universe="us")


def test_registered_universe_value_weights_align_to_returns():
    from goldman.data import load_returns

    path = config.value_weights_path("us")
    if path is None or not path.exists():
        pytest.skip("value_weights_us.csv not present")
    returns = load_returns(universe="us")
    weights = mc.load_value_weights(universe="us", assets=returns.columns, dates=returns.index)
    assert weights.index.equals(returns.index)
    assert weights.sum(axis=1).round(9).eq(1.0).all()