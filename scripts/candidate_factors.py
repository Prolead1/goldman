"""Every candidate factor we tried before choosing NEAR52 and VWML (report 3d).

Offline: uses only the committed data.  Run from the repo root::

    uv run python scripts/candidate_factors.py

Each signal is built on both universes with no look-ahead (every input is
known at the end of month ``t-1``) and sorted into the usual 30/40/30
equal-weighted long-short spread unless noted.  The table reports each
spread's annualised mean, its t-statistic and its CAPM alpha against the
market's own value-weighted index.  Ten signals and sixteen specifications in
Singapore (fifteen in the US, which has no dividend-yield sort), the count the
report's multiple-testing caveat refers to.  The momentum reference row is not
a candidate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from goldman import config
from goldman.analysis import build_custom_factors, factor_regression, long_short_spread
from goldman.data import load_returns, market_excess_returns, risk_free_series

warnings.filterwarnings("ignore")


def _rolling_market_fit(excess: pd.DataFrame, market: pd.Series, window: int):
    """Beta, residuals and residual std of each asset over ``t-window .. t-1``."""
    for i in range(window, len(excess)):
        y, x = excess.iloc[i - window : i], market.iloc[i - window : i]
        dx = x - x.mean()
        beta = y.sub(y.mean()).mul(dx, axis=0).sum() / (dx**2).sum()
        resid = y.sub(y.mean()) - np.outer(dx, beta)
        yield excess.index[i], beta, resid


def betting_against_beta(excess, market, window=24, shrink=0.6):
    """Frazzini-Pedersen: rank-weighted, each leg levered to beta one."""
    out = {}
    for date, beta, _ in _rolling_market_fit(excess, market, window):
        beta = shrink * beta + (1 - shrink)
        rank = beta.rank() - beta.rank().mean()
        high = rank.clip(lower=0) / rank.clip(lower=0).sum()
        low = (-rank).clip(lower=0) / (-rank).clip(lower=0).sum()
        r = excess.loc[date]
        out[date] = (low * r).sum() / (low * beta).sum() - (high * r).sum() / (high * beta).sum()
    return pd.Series(out)


def rolling_signal(excess, market, window, kind):
    rows = {}
    for date, beta, resid in _rolling_market_fit(excess, market, window):
        if kind == "beta":
            rows[date] = beta
        elif kind == "ivol":
            rows[date] = resid.std()
        elif kind == "resmom":  # Blitz, Huij and Martens: residuals t-12..t-2 / their std
            rows[date] = resid.iloc[-12:-1].sum() / resid.std()
    return pd.DataFrame(rows).T


def dividend_yield_sg(returns):
    """Trailing 12-month cash dividends over the month-end close, known at t-1."""
    yields = {}
    for ticker in returns.columns:
        path = config.RAW_DIR / "sg_daily" / f"{ticker}.csv"
        daily = pd.read_csv(path, index_col=0, parse_dates=True)
        month = daily.index.to_period("M")
        paid = daily["dividend"].fillna(0).groupby(month).sum().rolling(12).sum()
        yields[ticker] = paid / daily["close"].groupby(month).last()
    frame = pd.DataFrame(yields)
    frame.index = frame.index.to_timestamp()
    return frame.reindex(returns.index).shift(1)


def candidates(universe: str) -> dict[str, pd.Series]:
    returns = load_returns(universe=universe)
    excess = returns.sub(risk_free_series(universe, returns.index), axis=0)
    market = market_excess_returns(returns, universe=universe)
    custom, _ = build_custom_factors(returns)
    out = {f"BAB ({w}m beta)": betting_against_beta(excess, market, w) for w in (12, 24, 36, 60)}
    out["low beta"] = long_short_spread(
        returns, rolling_signal(excess, market, 24, "beta"), long_high=False
    )
    out["low volatility"] = long_short_spread(
        returns, returns.shift(1).rolling(24, min_periods=24).std(), long_high=False
    )
    out["low idiosyncratic vol"] = long_short_spread(
        returns, rolling_signal(excess, market, 24, "ivol"), long_high=False
    )
    out["low lottery (MAX)"] = long_short_spread(
        returns, returns.shift(1).rolling(12).max(), long_high=False
    )
    out["short-term reversal"] = long_short_spread(returns, returns.shift(1), long_high=False)
    out["residual momentum"] = long_short_spread(
        returns, rolling_signal(excess, market, 36, "resmom"), long_high=True
    )
    if universe == "sg":
        yields = dividend_yield_sg(returns)
        out["dividend yield"] = long_short_spread(returns, yields, long_high=True)
    out["52-week high (kept)"] = custom["NEAR52"]
    for q in (2, 4):
        variant, _ = build_custom_factors(returns, n_quantiles=q)
        out[f"52-week high, {q} groups"] = variant["NEAR52"]
    out["vol-managed momentum (kept)"] = custom["VWML"]
    variant, _ = build_custom_factors(returns, vol_window=12)
    out["vol-managed momentum, 12m window"] = variant["VWML"]
    out["momentum, WML (reference)"] = custom["WML"]
    return out, market


def main() -> int:
    rows = []
    for universe in ("us", "sg"):
        factors, market = candidates(universe)
        for name, series in factors.items():
            frame = pd.concat([series.rename("f"), market.rename("Mkt-RF")], axis=1).dropna()
            fit = factor_regression(frame[["f"]], frame[["Mkt-RF"]], rf=0.0).loc["f"]
            f = frame["f"]
            rows.append(
                {
                    "universe": universe,
                    "signal": name,
                    "months": len(f),
                    "ann. mean": f.mean() * 12,
                    "t(mean)": f.mean() / f.std() * np.sqrt(len(f)),
                    "CAPM alpha (ann.)": fit["alpha"] * 12,
                    "t(CAPM alpha)": fit["t_alpha"],
                }
            )
    table = pd.DataFrame(rows).set_index(["universe", "signal"])
    pd.set_option("display.width", 160)
    print(table.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
