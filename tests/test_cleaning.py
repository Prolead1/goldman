"""Tests for the Singapore total-return and market-cap rebuild."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldman import config
from goldman.data import cleaning as cl

HEADER = "ticker,ex_date,kind,ratio,distributed,distributed_price,start,end,source\n"


def _daily(sgx, factor_before=1.0, ex=None, dividends=None, currency="SGD", start="2020-01-01"):
    """A daily cache whose Yahoo close is SGX's price scaled by ``factor_before``
    on every day before ``ex`` (Yahoo's hidden adjustment)."""
    index = pd.bdate_range(start, periods=len(sgx))
    sgx = pd.Series(sgx, index=index, dtype=float)
    scale = pd.Series(1.0, index=index)
    if ex is not None:
        scale[index < pd.Timestamp(ex)] = factor_before
    frame = pd.DataFrame(
        {
            "close": sgx * scale,
            "adj_close": sgx * scale,
            "dividend": 0.0,
            "split": 0.0,
            "sgx_close": sgx,
            "currency": currency,
        }
    )
    for day, amount in (dividends or {}).items():
        frame.loc[pd.Timestamp(day), "dividend"] = amount
    return frame


def _actions(rows: str) -> pd.DataFrame:
    from io import StringIO

    return pd.read_csv(StringIO(HEADER + rows), parse_dates=["ex_date", "start", "end"])


def test_plain_total_return_adds_dividends():
    daily = _daily([10.0, 10.5, 10.2, 10.4], dividends={"2020-01-03": 0.2})
    index = cl.total_return_index(daily)
    expected = np.cumprod([1.0, 10.5 / 10.0, (10.2 + 0.2) / 10.5, 10.4 / 10.2])
    np.testing.assert_allclose(index.to_numpy(), expected)


def test_yahoo_factor_recovers_hidden_scaling():
    daily = _daily([2.0] * 6 + [1.0] * 6, factor_before=0.5, ex="2020-01-09")
    assert cl.yahoo_factor(daily, pd.Timestamp("2020-01-09")) == pytest.approx(0.5)


def test_yahoo_factor_rejects_thin_or_implausible_data():
    daily = _daily([2.0, 2.0, 1.0], factor_before=0.5, ex="2020-01-02")
    with pytest.raises(ValueError, match="Not enough"):
        cl.yahoo_factor(daily, pd.Timestamp("2019-12-01"))
    inflated = _daily([2.0] * 4 + [1.0] * 4, factor_before=2.0, ex="2020-01-07")
    with pytest.raises(ValueError, match="Implausible"):
        cl.yahoo_factor(inflated, pd.Timestamp("2020-01-07"))


def test_distribution_puts_earlier_dividends_on_the_scaled_basis():
    # Yahoo halves the close before the event but leaves the dividend raw.
    daily = _daily(
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        factor_before=0.5,
        ex="2020-01-08",
        dividends={"2020-01-02": 0.4},
    )
    actions = _actions("X,2020-01-08,distribution,0.1,units,,,,test\n")
    index = cl.total_return_index(daily, actions)
    # 0.4 raw on a 10.00 raw price is 4%, not the 8% Yahoo's mix implies.
    assert index.iloc[1] == pytest.approx(1.04)


def test_spinoff_ex_date_is_rebuilt_from_sgx_prices():
    daily = _daily([2.0] * 5 + [1.2] * 5, factor_before=0.5, ex="2020-01-08")
    actions = _actions("X,2020-01-08,spinoff,4.0,child,0.2,,,test\n")
    index = cl.total_return_index(daily, actions)
    i = daily.index.get_loc(pd.Timestamp("2020-01-08"))
    gross = index.iloc[i] / index.iloc[i - 1]
    assert gross == pytest.approx((1.2 + 4.0 * 0.2) / 2.0)  # holder kept 2.0 of value
    # Yahoo's own mix would have shown 1.2 / (0.5 * 2.0) = +20%.
    assert gross == pytest.approx(1.0)


def test_spinoff_needs_a_trading_day_and_prices():
    daily = _daily([2.0] * 5 + [1.2] * 5, factor_before=0.5, ex="2020-01-08")
    weekend = _actions("X,2020-01-11,spinoff,1,child,0.2,,,test\n")
    with pytest.raises(ValueError):
        cl.total_return_index(daily, weekend)
    gap = daily.copy()
    gap.loc[pd.Timestamp("2020-01-07"), "sgx_close"] = np.nan
    with pytest.raises(ValueError, match="Missing SGX"):
        cl.total_return_index(gap, _actions("X,2020-01-08,spinoff,1,child,0.2,,,test\n"))


def test_total_return_index_rejects_gaps():
    daily = _daily([10.0, 10.0, 10.0])
    daily.loc[daily.index[1], "close"] = np.nan
    with pytest.raises(ValueError, match="gaps"):
        cl.total_return_index(daily)


def test_month_end_labels_by_month_start():
    days = pd.to_datetime(["2020-01-15", "2020-01-31", "2020-02-03"])
    series = pd.Series([1.0, 2.0, 3.0], index=days)
    monthly = cl.month_end(series)
    assert list(monthly.index) == list(pd.to_datetime(["2020-01-01", "2020-02-01"]))
    assert monthly.tolist() == [2.0, 3.0]


def test_clean_shares_carries_the_last_good_count():
    index = pd.date_range("2020-01-01", periods=5, freq="MS")
    shares = pd.Series([100.0, 1.0, 900.0, 100.0, 110.0], index=index)
    actions = _actions("X,,shares_carry,,,,2020-02-01,2020-03-01,glitch\n")
    cleaned = cl.clean_shares(shares, actions)
    assert cleaned.tolist() == [100.0, 100.0, 100.0, 100.0, 110.0]
    assert cl.clean_shares(shares).equals(shares)
    early = _actions("X,,shares_carry,,,,2020-01-01,2020-02-01,glitch\n")
    with pytest.raises(ValueError, match="carry forward"):
        cl.clean_shares(shares, early)


def test_monthly_market_caps_use_sgx_price_shares_and_fx():
    daily = _daily([10.0] * 25 + [12.0] * 20, currency="USD")
    shares = pd.Series([5.0], index=pd.to_datetime(["2019-12-01"]))
    fx = pd.Series([1.3, 1.4, 1.5], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    caps = cl.monthly_market_caps(daily, shares, fx)
    assert caps.loc["2020-01-01"] == pytest.approx(10.0 * 5 * 1.3)
    assert caps.loc["2020-03-01"] == pytest.approx(12.0 * 5 * 1.5)


def test_listing_currency_defaults_to_sgd():
    assert cl.listing_currency(_daily([1.0, 1.0], currency="USD")) == "USD"
    assert cl.listing_currency(_daily([1.0, 1.0]).drop(columns="currency")) == "SGD"


def test_load_corporate_actions_validates(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("# comment\n" + HEADER + "X,2020-01-08,spinoff,1,child,0.2,,,ok\n")
    assert len(cl.load_corporate_actions(good)) == 1
    unknown = tmp_path / "unknown.csv"
    unknown.write_text(HEADER + "X,2020-01-08,merger,1,child,0.2,,,bad\n")
    with pytest.raises(ValueError, match="Unknown"):
        cl.load_corporate_actions(unknown)
    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text(HEADER + "X,2020-01-08,spinoff,,child,,,,bad\n")
    with pytest.raises(ValueError, match="ratio"):
        cl.load_corporate_actions(incomplete)


def test_build_sg_clean_end_to_end(tmp_path):
    a = _daily(np.linspace(10, 12, 60), start="2020-01-01")
    b = _daily(np.linspace(5, 4, 60), currency="USD", start="2020-01-01")
    a.to_csv(tmp_path / "A.SI.csv")
    b.to_csv(tmp_path / "B.SI.csv")
    fx = pd.Series([1.30, 1.35, 1.40], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    shares = {
        "A.SI": pd.Series([100.0], index=pd.to_datetime(["2019-12-01"])),
        "B.SI": pd.Series([10.0], index=pd.to_datetime(["2019-12-01"])),
    }
    result = cl.build_sg_clean(
        start="2020-01-01",
        raw_dir=tmp_path,
        actions=_actions("A.SI,,shares_carry,,,,2020-02-01,2020-02-01,test\n"),
        fx=fx,
        shares=shares,
    )
    assert list(result.prices.columns) == ["A.SI", "B.SI"]
    # The USD listing is converted: its SGD return compounds the FX move.
    usd_local = cl.month_end(b["sgx_close"])
    expected = (usd_local * fx).pct_change().dropna()
    np.testing.assert_allclose(result.prices["B.SI"].pct_change().dropna(), expected)
    totals = result.value_weights.groupby("date")["weight"].sum()
    np.testing.assert_allclose(totals, 1.0)
    assert "B.SI" in result.report and "carried forward" in result.report
    assert result.summary.loc["B.SI", "currency"] == "USD"


# --------------------------------------------------- the committed SG files
@pytest.fixture(scope="module")
def rebuilt():
    if not (config.RAW_DIR / cl.DAILY_DIR).is_dir():
        pytest.skip("SG daily cache not present")
    return cl.build_sg_clean()


def test_committed_sg_files_reproduce_from_the_raw_cache(rebuilt):
    prices = pd.read_csv(config.data_path("prices_sg_monthly.csv"), index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(
        rebuilt.prices[prices.columns], prices, check_freq=False, check_names=False, rtol=1e-9
    )
    weights = pd.read_csv(config.data_path("value_weights_sg.csv"), parse_dates=["date"])
    np.testing.assert_allclose(rebuilt.value_weights["weight"], weights["weight"], rtol=1e-9)


def test_committed_sg_prices_have_no_impossible_moves(rebuilt):
    returns = rebuilt.prices.pct_change()
    assert returns.abs().max().max() < 0.6  # every month beyond 30% is a known real move
    growth = rebuilt.summary["growth clean (local)"]
    assert growth["Y92.SI"] < 1.0  # ThaiBev lost money; Yahoo said 287x
    assert growth.max() < 25.0
