"""Tests for the value-weight (market-cap) loader."""

from __future__ import annotations

import pandas as pd
import pytest

from goldman import config
from goldman.data.market_caps import load_value_weights, value_weights_from_market_caps


def test_value_weights_from_market_caps_normalises():
    caps = pd.Series({"a": 100.0, "b": 300.0})
    weights = value_weights_from_market_caps(caps)
    assert weights["a"] == pytest.approx(0.25)
    assert weights.sum() == pytest.approx(1.0)


def test_value_weights_reject_missing_or_nonpositive_total():
    with pytest.raises(ValueError):
        value_weights_from_market_caps(pd.Series({"a": 1.0, "b": float("nan")}))
    with pytest.raises(ValueError):
        value_weights_from_market_caps(pd.Series({"a": 0.0, "b": 0.0}))


def test_load_value_weights_aligns_to_asset_order(tmp_path):
    path = tmp_path / "value_weights.csv"
    pd.DataFrame({"asset": ["a", "b"], "market_cap": [100.0, 100.0]}).to_csv(path, index=False)

    weights = load_value_weights(path=path, assets=["b", "a"])
    assert list(weights.index) == ["b", "a"]
    assert weights["a"] == pytest.approx(0.5)
    assert weights.sum() == pytest.approx(1.0)


def test_load_value_weights_from_weight_column_and_percent(tmp_path):
    path = tmp_path / "value_weights.csv"
    pd.DataFrame({"asset": ["a", "b"], "weight": [25.0, 75.0]}).to_csv(path, index=False)

    weights = load_value_weights(path=path, assets=["a", "b"], as_percent=True)
    assert weights["a"] == pytest.approx(0.25)


def test_load_value_weights_raises_when_asset_uncovered(tmp_path):
    path = tmp_path / "value_weights.csv"
    pd.DataFrame({"asset": ["a"], "weight": [1.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing"):
        load_value_weights(path=path, assets=["a", "b"])


def test_load_value_weights_requires_path_or_universe():
    with pytest.raises(ValueError):
        load_value_weights()
    with pytest.raises(ValueError):
        load_value_weights(path="x.csv", universe="us")


def test_registered_universe_value_weights_align_to_returns():
    from goldman.data import load_returns

    path = config.value_weights_path("us")
    if path is None or not path.exists():
        pytest.skip("value_weights_us.csv not present")
    returns = load_returns(universe="us")
    weights = load_value_weights(universe="us", assets=returns.columns)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.index.equals(returns.columns)