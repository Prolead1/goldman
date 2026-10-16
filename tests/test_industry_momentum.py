"""Tests for industry momentum on the Ken French 10 industries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goldman import viz
from goldman.analysis.factor_construction import momentum_factor
from goldman.analysis.industry_momentum import build_industry_momentum, industry_momentum_report
from goldman.data import load_long_history


def _industries(seed: int = 0, n_obs: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.period_range("1990-01", periods=n_obs, freq="M")
    data = rng.normal(0.008, 0.05, (n_obs, 10))
    return pd.DataFrame(data, index=index, columns=[f"ind{i}" for i in range(10)])


def _factors(index: pd.PeriodIndex, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.005, 0.04, (len(index), 4))
    return pd.DataFrame(data, index=index, columns=["Mkt-RF", "SMB", "HML", "Mom"])


def test_industry_momentum_is_wml_with_three_per_leg():
    industries = _industries()
    factor, traded = build_industry_momentum(industries)
    dated = industries.copy()
    dated.index = dated.index.to_timestamp()
    wml = momentum_factor(dated).dropna()
    pd.testing.assert_series_equal(factor, wml, check_names=False, check_freq=False)
    # The 12-2 signal needs 12 months of history, so the factor starts in month 13.
    assert factor.index[0] == dated.index[12]
    assert np.isclose(traded.iloc[0], 2.0)  # first month buys both legs in full
    assert (traded <= 4.0 + 1e-12).all()


def test_report_rows_and_columns():
    industries = _industries()
    factor, traded = build_industry_momentum(industries)
    factors = _factors(industries.index)
    periods = {"all": ("1991-01", "1999-12"), "late": ("1995-01", "1999-12")}
    table = industry_momentum_report(factor, traded, factors, periods=periods)
    assert list(table.index) == ["all", "late"]
    assert table.loc["all", "months"] == 108
    assert table.loc["late", "months"] == 60
    for column in ("FF3 alpha (ann.)", "corr with Mom", "break-even cost (bp)"):
        assert np.isfinite(table[column].astype(float)).all()


def test_committed_history_reproduces_the_published_result():
    # Regression lock on the committed vintage (Ken French files built from CRSP
    # 202608). A newer download moves these numbers and should update the report.
    history = load_long_history()
    assert history.industries.shape[1] == 10
    assert str(history.industries.index[0]) == "1963-07"
    factor, traded = build_industry_momentum(history.industries)
    assert f"{factor.index[0]:%Y-%m}" == "1964-07"
    table = industry_momentum_report(
        factor, traded, history.factors, periods={"full": ("1964-07", "2026-08")}
    )
    row = table.loc["full"]
    assert row["months"] == 746
    assert abs(row["ann. mean"] - 0.038) < 0.001
    assert abs(row["corr with Mom"] - 0.72) < 0.01


def test_plot_industry_momentum(tmp_path):
    industries = _industries(n_obs=200)
    factor, _ = build_industry_momentum(industries)
    factors = _factors(industries.index)
    factors.index = factors.index.to_timestamp()
    panel = factors[["Mom", "Mkt-RF"]].reindex(factor.index).assign(IndMom=factor)
    path = viz.plot_industry_momentum(
        panel, tmp_path / "ind.png", window=60, highlight=("2000-01", "2005-12")
    )
    assert path.exists() and path.stat().st_size > 0
