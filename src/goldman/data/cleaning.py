"""Singapore total returns and market caps rebuilt from exchange data.

Yahoo's adjusted close is wrong for several SGX names.  An audit against SGX's
own unadjusted prices and corporate-action records found:

* **ThaiBev (Y92).** Before 2024 Yahoo adjusted every dividend using the baht
  amount as if it were Singapore dollars, about 26 times too large, so the
  adjusted series grows 287x where the true total return is about 0.65x.
* **Jardine C&C (C07).** Since 2020-09 Yahoo adjusts with the US-dollar
  amount, so each dividend counts at about 78% of its SGD value.
* **Distributions in specie (U96, BS6, BN4, C09).** Yahoo scales the history
  of ``Close`` by a price-ratio factor but leaves earlier dividends unscaled,
  which inflates pre-event dividend yields by up to 2x.  For the three large
  spin-offs the factor also turns the ex-date into a +13% to +22% jump where
  holders actually gained +3% to +10%.
* **Market caps.** The scaled ``Close`` understates pre-event caps (Sembcorp
  by half), Yahoo's share counts glitch (UOB in 2017-03, Jardine Matheson from
  2019-07 to 2022-05), and the three US-dollar listings were left in USD.

The rebuild therefore uses, per ticker:

* **Returns.** Daily ``(close + dividend) / previous close`` from Yahoo's
  split- and capital-adjusted close.  Dividends paid before a distribution in
  specie are rescaled onto the close's basis.  Each large spin-off's ex-date
  return is replaced by what a holder actually earned: (SGX ex-date price +
  cash dividend + distributed shares at their SGX price) / SGX cum price.
* **Market caps.** SGX's unadjusted month-end price times Yahoo's
  point-in-time share count, with the documented share-count overrides.
* **Currency.** US-dollar listings are converted to SGD at month-end USD/SGD,
  so every Singapore series is an SGD total return.

Every correction is a row of ``data/corporate_actions_sg.csv`` with its source,
and :func:`build_sg_clean` reports what each one changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .market_caps import long_value_weights

__all__ = [
    "SPINOFF",
    "DISTRIBUTION",
    "SHARES_CARRY",
    "load_corporate_actions",
    "load_sg_daily",
    "load_fx",
    "listing_currency",
    "yahoo_factor",
    "total_return_index",
    "month_end",
    "clean_shares",
    "monthly_market_caps",
    "SGCleanResult",
    "build_sg_clean",
]

SPINOFF = "spinoff"  # large distribution in specie: ex-date return rebuilt
DISTRIBUTION = "distribution"  # small distribution in specie: Yahoo's price factor kept
SHARES_CARRY = "shares_carry"  # share count carried forward across a glitch window
KINDS = {SPINOFF, DISTRIBUTION, SHARES_CARRY}

CORPORATE_ACTIONS_FILE = "corporate_actions_sg.csv"
DAILY_DIR = "sg_daily"
FX_FILE = "fx/usdsgd_monthly.csv"
SG_START = "2016-09-01"  # first month of the committed SG panel


# --------------------------------------------------------------------- inputs
def load_corporate_actions(path: str | Path | None = None) -> pd.DataFrame:
    """Read the curated corporate-action corrections (one row per event)."""
    csv = Path(path) if path is not None else config.data_path(CORPORATE_ACTIONS_FILE)
    actions = pd.read_csv(csv, parse_dates=["ex_date", "start", "end"], comment="#")
    unknown = set(actions["kind"]) - KINDS
    if unknown:
        raise ValueError(f"Unknown corporate-action kind(s): {', '.join(sorted(unknown))}")
    spinoffs = actions[actions["kind"] == SPINOFF]
    if spinoffs[["ratio", "distributed_price"]].isna().any().any():
        raise ValueError("Every spin-off needs a ratio and a distributed_price.")
    return actions


def load_sg_daily(ticker: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Read one ticker's daily cache: Yahoo close/dividend next to SGX's price."""
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DIR / DAILY_DIR
    frame = pd.read_csv(directory / f"{ticker}.csv", index_col=0, parse_dates=True)
    return frame.sort_index()


def load_fx(path: str | Path | None = None) -> pd.Series:
    """Month-end USD/SGD (SGD per USD), labelled by month start."""
    csv = Path(path) if path is not None else config.RAW_DIR / FX_FILE
    return pd.read_csv(csv, index_col=0, parse_dates=True)["usdsgd"].sort_index()


def listing_currency(daily: pd.DataFrame) -> str:
    """The currency SGX quotes the stock in (most common reported value)."""
    currency = daily["currency"].dropna() if "currency" in daily else pd.Series(dtype=str)
    return str(currency.mode().iloc[0]) if len(currency) else "SGD"


# -------------------------------------------------------------------- returns
def yahoo_factor(daily: pd.DataFrame, ex_date: pd.Timestamp, window: int = 5) -> float:
    """Yahoo's hidden price factor at ``ex_date``: its close / SGX's, before / after.

    Yahoo scales the whole history of ``close`` before a distribution in specie
    by a constant.  The ratio of Yahoo's close to SGX's unadjusted price is
    therefore flat on each side of the event; the median of ``window`` days on
    each side gives the factor without being thrown by one stale quote.
    """
    ratio = (daily["close"] / daily["sgx_close"]).replace([np.inf, -np.inf], np.nan).dropna()
    before = ratio[ratio.index < ex_date].tail(window)
    after = ratio[ratio.index >= ex_date].head(window)
    if before.empty or after.empty:
        raise ValueError(f"Not enough SGX prices around {ex_date:%Y-%m-%d} to measure the factor.")
    factor = float(before.median() / after.median())
    if not 0.0 < factor <= 1.0 + 1e-3:
        raise ValueError(f"Implausible Yahoo factor {factor:.4f} at {ex_date:%Y-%m-%d}.")
    return factor


def total_return_index(daily: pd.DataFrame, actions: pd.DataFrame | None = None) -> pd.Series:
    """Daily total-return index in the listing currency, starting at 1.

    ``actions`` holds this ticker's rows of the corporate-action file.  A
    dividend paid before a distribution in specie is multiplied by Yahoo's
    factor for that event, which puts it on the same basis as the scaled
    close.  On a spin-off's ex-date the return is rebuilt from SGX prices:
    ``(ex price + cash dividend + ratio x distributed price) / cum price``.
    """
    close = daily["close"].astype(float)
    dividend = daily["dividend"].fillna(0.0).astype(float)
    actions = actions if actions is not None else pd.DataFrame(columns=["kind", "ex_date"])
    events = actions[actions["kind"].isin([SPINOFF, DISTRIBUTION])]

    basis = pd.Series(1.0, index=daily.index)
    for event in events.itertuples():
        basis[daily.index < event.ex_date] *= yahoo_factor(daily, event.ex_date)

    gross = (close + dividend * basis) / close.shift(1)
    for event in events[events["kind"] == SPINOFF].itertuples():
        if event.ex_date not in daily.index:
            raise ValueError(f"Spin-off ex-date {event.ex_date:%Y-%m-%d} is not a trading day.")
        i = daily.index.get_loc(event.ex_date)
        cum, ex = daily["sgx_close"].iloc[i - 1], daily["sgx_close"].iloc[i]
        if not (np.isfinite(cum) and np.isfinite(ex)):
            raise ValueError(f"Missing SGX price around {event.ex_date:%Y-%m-%d}.")
        distributed = float(event.ratio) * float(event.distributed_price)
        gross.iloc[i] = (ex + dividend.iloc[i] + distributed) / cum

    gross.iloc[0] = 1.0
    if gross.isna().any():
        raise ValueError("Total-return index has gaps; check the daily cache.")
    return gross.cumprod().rename(daily.index.name or "index")


def month_end(series: pd.Series) -> pd.Series:
    """Last value of each month, labelled by the month's first day."""
    monthly = series.groupby(series.index.to_period("M")).last()
    monthly.index = monthly.index.to_timestamp()
    return monthly


# ---------------------------------------------------------------- market caps
def clean_shares(shares: pd.Series, actions: pd.DataFrame | None = None) -> pd.Series:
    """Carry the last good share count across each documented glitch window."""
    shares = shares.astype(float).copy()
    if actions is None:
        return shares
    for event in actions[actions["kind"] == SHARES_CARRY].itertuples():
        good = shares[shares.index < event.start].dropna()
        if good.empty:
            raise ValueError(f"No share count before {event.start:%Y-%m-%d} to carry forward.")
        shares[(shares.index >= event.start) & (shares.index <= event.end)] = good.iloc[-1]
    return shares


def monthly_market_caps(
    daily: pd.DataFrame,
    shares: pd.Series,
    fx: pd.Series | None = None,
) -> pd.Series:
    """Point-in-time market cap in SGD: SGX month-end price x shares (x FX).

    Shares are forward-filled from their report dates onto the monthly index,
    with the earliest known count carried back where Yahoo has no early
    history (the same convention as :func:`value_weights_from_raw`).
    """
    price = month_end(daily["sgx_close"].ffill()).dropna()
    aligned = shares.reindex(shares.index.union(price.index)).ffill().bfill().reindex(price.index)
    cap = price * aligned
    if fx is not None:
        cap = cap * fx.reindex(cap.index)
    return cap.rename("market_cap")


# ---------------------------------------------------------------------- build
@dataclass
class SGCleanResult:
    """Clean SG inputs plus a plain-text account of every correction."""

    prices: pd.DataFrame
    value_weights: pd.DataFrame
    report: str
    summary: pd.DataFrame


def _shares(ticker: str) -> pd.Series:
    path = config.raw_universe_dir("sg") / f"{ticker}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)["shares"]


def build_sg_clean(
    tickers: list[str] | None = None,
    *,
    start: str = SG_START,
    raw_dir: str | Path | None = None,
    actions: pd.DataFrame | None = None,
    fx: pd.Series | None = None,
    shares: dict[str, pd.Series] | None = None,
) -> SGCleanResult:
    """Rebuild the SG monthly total-return prices and value weights.

    ``prices`` has one column per ticker in SGD, scaled so the last month equals
    the last SGX price (only returns are used downstream).  ``value_weights``
    is the long ``asset, date, market_cap, weight`` layout of
    ``data/value_weights_sg.csv``.
    """
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DIR / DAILY_DIR
    if tickers is None:
        tickers = sorted(p.stem for p in directory.glob("*.csv"))
    actions = actions if actions is not None else load_corporate_actions()
    fx = fx if fx is not None else load_fx()

    prices, caps, rows, lines = {}, {}, [], []
    for ticker in tickers:
        daily = load_sg_daily(ticker, directory)
        acts = actions[actions["ticker"] == ticker]
        currency = listing_currency(daily)
        to_sgd = fx if currency == "USD" else None

        local = month_end(total_return_index(daily, acts))
        level = local * (to_sgd.reindex(local.index) if to_sgd is not None else 1.0)
        last_price = month_end(daily["sgx_close"].ffill()).iloc[-1]
        last_price *= to_sgd.reindex(local.index).iloc[-1] if to_sgd is not None else 1.0
        prices[ticker] = level / level.iloc[-1] * last_price

        ticker_shares = shares[ticker] if shares is not None else _shares(ticker)
        caps[ticker] = monthly_market_caps(daily, clean_shares(ticker_shares, acts), to_sgd)

        yahoo = month_end(daily["adj_close"]).loc[start:]
        clean = local.loc[start:]
        gap = (clean.pct_change() - yahoo.pct_change()).abs()
        rows.append(
            {
                "ticker": ticker,
                "currency": currency,
                "months": int(clean.notna().sum()),
                "growth Yahoo adj (local)": yahoo.iloc[-1] / yahoo.dropna().iloc[0],
                "growth clean (local)": clean.iloc[-1] / clean.dropna().iloc[0],
                "max monthly gap": gap.max(),
                "months gap > 1%": int((gap > 0.01).sum()),
            }
        )
        for event in acts.itertuples():
            if event.kind == SHARES_CARRY:
                lines.append(
                    f"{ticker}: share count {event.start:%Y-%m}..{event.end:%Y-%m} carried "
                    f"forward ({event.source})"
                )
                continue
            month = event.ex_date.to_period("M").to_timestamp()
            after = clean.pct_change().get(month, np.nan)
            before = yahoo.pct_change().get(month, np.nan)
            lines.append(
                f"{ticker}: {event.kind} {event.ex_date:%Y-%m-%d}, Yahoo factor "
                f"{yahoo_factor(daily, event.ex_date):.4f}; {month:%Y-%m} return "
                f"{before:+.1%} (Yahoo) -> {after:+.1%} (clean) ({event.source})"
            )

    price_panel = pd.DataFrame(prices).sort_index().loc[start:]
    price_panel.index.name = "Date"
    cap_panel = pd.DataFrame(caps).sort_index()
    weights = long_value_weights(cap_panel)
    summary = pd.DataFrame(rows).set_index("ticker")

    usd = summary.index[summary["currency"] == "USD"].tolist()
    big = price_panel.pct_change().stack()
    big = big[big.abs() > 0.30]
    report = "\n".join(
        [
            "Singapore data cleaning report (goldman.data.cleaning.build_sg_clean)",
            "",
            "Returns: Yahoo daily close + dividends, dividends rescaled before each",
            "distribution in specie, spin-off ex-dates rebuilt from SGX prices.",
            "Market caps: SGX unadjusted month-end price x point-in-time shares.",
            f"Converted from USD to SGD at month-end USD/SGD: {', '.join(usd) or 'none'}.",
            "",
            "Per-ticker comparison with Yahoo's adjusted close (local currency):",
            summary.round(4).to_string(),
            "",
            "Corrections applied:",
            *lines,
            "",
            "Clean monthly returns beyond +/-30% (check each is a real move):",
            big.round(4).to_string() if len(big) else "none",
            "",
        ]
    )
    return SGCleanResult(prices=price_panel, value_weights=weights, report=report, summary=summary)
