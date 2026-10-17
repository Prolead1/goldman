"""Tests for the self-built size and momentum factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman.analysis.factor_construction import (
    build_self_factors,
    long_short_spread,
    long_short_weights,
    momentum_factor,
    momentum_signal,
    momentum_tradability,
    self_factor_tradability,
    size_factor,
    turnover,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2010-01-01", periods=n, freq="MS")


def _row(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame([values], index=_dates(1), columns=list("abcdef"))


def test_long_short_spread_longs_the_top_signal():
    returns = _row([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    signal = _row([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spread = long_short_spread(returns, signal, n_quantiles=3)
    # Top third e, f; bottom third a, b: mean(0.05, 0.06) - mean(0.01, 0.02).
    assert spread.iloc[0] == pytest.approx(0.04)


def test_long_short_spread_can_long_low_signal():
    returns = _row([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    signal = _row([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spread = long_short_spread(returns, signal, n_quantiles=3, long_high=False)
    assert spread.iloc[0] == pytest.approx(-0.04)


def test_long_short_spread_returns_nan_with_too_few_assets():
    dates = _dates(1)
    returns = pd.DataFrame([[0.01, 0.02]], index=dates, columns=list("ab"))
    signal = pd.DataFrame([[1.0, 2.0]], index=dates, columns=list("ab"))
    spread = long_short_spread(returns, signal, n_quantiles=3)
    assert spread.isna().all()


def test_long_short_spread_rejects_bad_quantiles():
    with pytest.raises(ValueError, match="at least 2"):
        long_short_spread(pd.DataFrame([[0.1]]), pd.DataFrame([[1.0]]), n_quantiles=1)


def test_size_factor_rewards_small_caps_and_lags_the_weights():
    dates = _dates(6)
    columns = list("abcdef")
    # Small caps (low weight) earn more every month.
    monthly = np.linspace(0.06, 0.01, len(columns))
    returns = pd.DataFrame(np.tile(monthly, (len(dates), 1)), index=dates, columns=columns)
    weights = pd.DataFrame(
        np.tile(np.arange(1.0, len(columns) + 1), (len(dates), 1)), index=dates, columns=columns
    ).div(np.arange(1.0, len(columns) + 1).sum())
    factor = size_factor(returns, weights, n_quantiles=3)
    # The first month has no lagged weight, so it is undefined.
    assert np.isnan(factor.iloc[0])
    assert (factor.iloc[1:] > 0).all()


def test_momentum_signal_is_lagged_formation_return():
    dates = _dates(20)
    rate = 0.02
    returns = pd.DataFrame(rate, index=dates, columns=["a"])
    signal = momentum_signal(returns, lookback=12, skip=1)
    # First valid month is when months t-12..t-2 exist: index 12.
    assert signal.iloc[:12].isna().all().all()
    assert signal.iloc[12, 0] == pytest.approx((1 + rate) ** 11 - 1.0)


def test_momentum_factor_rewards_continuing_winners():
    dates = _dates(20)
    columns = list("abcdef")
    monthly = np.linspace(0.01, 0.06, len(columns))
    returns = pd.DataFrame(np.tile(monthly, (len(dates), 1)), index=dates, columns=columns)
    factor = momentum_factor(returns, lookback=12, skip=1)
    assert factor.iloc[:12].isna().all()
    assert (factor.iloc[12:] > 0).all()


def test_momentum_requires_lookback_beyond_skip():
    returns = pd.DataFrame(0.01, index=_dates(20), columns=["a"])
    with pytest.raises(ValueError, match="lookback must exceed"):
        momentum_signal(returns, lookback=1, skip=1)


def test_build_self_factors_columns_and_requires_source():
    dates = _dates(20)
    returns = pd.DataFrame(
        np.tile(np.linspace(0.01, 0.05, 5), (len(dates), 1)), index=dates, columns=list("abcde")
    )
    weights = pd.DataFrame(
        np.tile(np.linspace(0.1, 0.3, 5), (len(dates), 1)), index=dates, columns=list("abcde")
    )
    factors = build_self_factors(returns, value_weights=weights)
    assert list(factors.columns) == ["SMB", "WML"]
    assert factors.index.equals(returns.index)
    with pytest.raises(ValueError, match="value_weights"):
        build_self_factors(returns)


def test_long_short_weights_reproduce_the_spread():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0.008, 0.06, (60, 9)), index=_dates(60),
                           columns=[f"a{i}" for i in range(9)])
    signal = returns.shift(1)
    weights = long_short_weights(returns, signal, n_quantiles=3)
    spread = long_short_spread(returns, signal, n_quantiles=3)
    rebuilt = (weights * returns).sum(axis=1).where(weights.abs().sum(axis=1) > 0)
    pd.testing.assert_series_equal(rebuilt, spread, check_names=False, check_freq=False)
    active = weights.abs().sum(axis=1) > 0
    np.testing.assert_allclose(weights[active].sum(axis=1), 0.0, atol=1e-12)  # dollar neutral
    with pytest.raises(ValueError):
        long_short_weights(returns, signal, n_quantiles=1)


def test_turnover_counts_formation_and_flips():
    weights = pd.DataFrame(
        [[0.5, -0.5], [-0.5, 0.5]], index=_dates(2), columns=["a", "b"]
    )
    traded = turnover(weights)
    # Month one opens 1.0 gross; month two flips both half positions, so 2.0.
    assert traded.iloc[0] == pytest.approx(1.0)
    assert traded.iloc[1] == pytest.approx(2.0)


def test_momentum_tradability_matches_the_reported_factor():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        rng.normal(0.008, 0.06, (80, 9)), index=_dates(80), columns=[f"a{i}" for i in range(9)]
    )
    result = momentum_tradability(returns)
    expected = momentum_factor(returns).dropna()
    # The tradability factor is exactly the WML Stage 3 reports.
    pd.testing.assert_series_equal(
        result.factor, expected, check_names=False, check_freq=False
    )
    assert (result.turnover >= 0).all()
    assert (result.drawdown <= 1e-12).all()
    # At zero cost the curve is the gross annualised mean; at break-even it is zero.
    assert result.cost_curve([0]).loc[0] == pytest.approx(result.factor.mean() * 12)
    break_even = result.cost_curve([result.break_even_bps]).loc[result.break_even_bps]
    assert break_even == pytest.approx(0.0, abs=1e-12)
    assert result.break_even_bps == pytest.approx(
        result.factor.mean() / result.turnover.mean() * 1e4
    )


def test_self_factor_tradability_matches_smb_and_wml():
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(
        rng.normal(0.008, 0.06, (80, 9)), index=_dates(80), columns=[f"a{i}" for i in range(9)]
    )
    value_weights = pd.DataFrame(
        rng.uniform(0.05, 0.3, (80, 9)), index=_dates(80), columns=[f"a{i}" for i in range(9)]
    )
    built = build_self_factors(returns, value_weights=value_weights)
    results = self_factor_tradability(returns, value_weights)
    assert set(results) == {"SMB", "WML"}
    for factor in ("SMB", "WML"):
        pd.testing.assert_series_equal(
            results[factor].factor,
            built[factor].dropna(),
            check_names=False,
            check_freq=False,
        )
        assert results[factor].break_even_bps == pytest.approx(
            results[factor].factor.mean() / results[factor].turnover.mean() * 1e4
        )
