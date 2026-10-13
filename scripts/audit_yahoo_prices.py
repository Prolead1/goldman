"""Audit the committed price files against Yahoo's raw closes and dividends.

For every ticker in ``data/prices_<universe>_monthly.csv`` this rebuilds the
monthly total return as ``(close + dividends) / previous close`` from Yahoo's
daily data, compares it with the committed series, and checks each ex-date's
adjustment factor against the correct ``1 - dividend / previous close``.  A
clean series agrees to a few basis points a month; a broken one shows up as a
large gap or a flagged ex-date.

Network only (needs the ``data`` extra).  Run from the repo root::

    uv run python scripts/audit_yahoo_prices.py            # US and SG
    uv run python scripts/audit_yahoo_prices.py --universe sg

For the SG file, gaps are expected exactly where ``goldman.data.cleaning``
corrected Yahoo (see ``data/cleaning_report_sg.txt``) and for the three
US-dollar listings, which the committed file holds in SGD.

Note: this check cannot see a spin-off.  Yahoo builds those into its raw
close as well, so both sides of the comparison carry the error; the SG
rebuild in ``goldman.data.cleaning`` compares against SGX's own prices for
that reason.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from goldman import config


def audit_ticker(ticker: str, committed: pd.Series) -> dict:
    import yfinance as yf

    handle = yf.Ticker(ticker)
    daily = handle.history(start="2016-06-01", end="2026-09-01", auto_adjust=False, actions=True)
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    daily = daily[daily["Close"] > 0]

    flagged = 0
    for date, dividend in daily.loc[daily["Dividends"] > 0, "Dividends"].items():
        i = daily.index.get_loc(date)
        if i == 0:
            continue
        before, after = daily.iloc[i - 1], daily.iloc[i]
        yahoo = (before["Adj Close"] / before["Close"]) / (after["Adj Close"] / after["Close"])
        flagged += abs(yahoo - (1 - dividend / before["Close"])) > 0.005

    gross = ((daily["Close"] + daily["Dividends"]) / daily["Close"].shift(1)).dropna()
    rebuilt = gross.groupby(gross.index.to_period("M")).prod() - 1
    returns = committed.dropna().pct_change().dropna()
    returns.index = returns.index.to_period("M")
    common = returns.index.intersection(rebuilt.index)
    gap = (returns[common] - rebuilt[common]).abs()
    return {
        "ticker": ticker,
        "currency": handle.history_metadata.get("currency"),
        "months": len(common),
        "flagged ex-dates": int(flagged),
        "max monthly gap": float(gap.max()),
        "months gap > 1%": int((gap > 0.01).sum()),
        "growth committed": float((1 + returns[common]).prod()),
        "growth rebuilt": float((1 + rebuilt[common]).prod()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=["us", "sg"], default=None)
    parser.add_argument("--pace", type=float, default=0.5)
    args = parser.parse_args()

    universes = [args.universe] if args.universe else ["us", "sg"]
    rows = []
    for universe in universes:
        prices = pd.read_csv(config.universe_path(universe), index_col=0, parse_dates=True)
        for ticker in prices.columns:
            rows.append({"universe": universe, **audit_ticker(ticker, prices[ticker])})
            time.sleep(args.pace)
    table = pd.DataFrame(rows).set_index(["universe", "ticker"])
    pd.set_option("display.width", 200)
    print(table.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
