"""Explicit, network-only fetchers — never imported by the analysis pipeline.

Everything that touches the internet lives here and runs **only when invoked**;
importing ``goldman.data`` (or running the notebook) never calls it.
The optional ``data`` extra (yfinance, pandas-datareader) is imported lazily
inside each function, so the rest of the package works without it.

Usage::

    uv sync --extra data
    uv run python -m goldman.data.fetch prices       # prices + FF3 factors
    uv run python -m goldman.data.fetch caps         # fetch + build value weights
    uv run python -m goldman.data.fetch caps-fetch   # network -> raw per-ticker cache
    uv run python -m goldman.data.fetch caps-build   # raw cache -> value weights (offline)
    uv run python -m goldman.data.fetch sg-daily     # network -> SG daily raw cache + FX
    uv run python -m goldman.data.fetch sg-build     # SG daily cache -> clean SG files (offline)
    uv run python -m goldman.data.fetch long-history # Ken French 10 industries + factors since 1926
    uv run python -m goldman.data.fetch all          # everything

Market caps are point-in-time: ``split-unadjusted close(t) x shares(t)`` with
shares forward-filled from their report dates, so the value-weighted benchmark
never applies today's market cap to eight-year-old data.

Design notes
------------
* Prices: ONE batch call per universe, then cached; a per-ticker Stooq fallback
  covers names Yahoo misses (Yahoo rate-limits repeated pulls).
* Market caps: one pass per ticker (prices + shares together) into a raw cache
  under ``data/raw/<universe>/``; ``caps-build`` is offline and
  repeatable.  ``--refresh`` re-pulls, ``--pace`` controls politeness.
* US prices are Yahoo's adjusted close (splits + dividends) — a total-return
  proxy that passed our audit against raw closes plus dividends.
* SG prices are *not* Yahoo's adjusted close, which is wrong for several SGX
  names (see ``goldman.data.cleaning``).  ``sg-daily`` caches Yahoo's daily
  close and dividends next to SGX's own unadjusted last-traded price, and
  ``sg-build`` rebuilds total returns and market caps from that cache offline.
* The SG list is "approximately the STI": verify current membership before
  relying on the value-weight comparison.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .. import config
from .cleaning import build_sg_clean
from .market_caps import build_value_weights_from_raw, unadjust_close

__all__ = ["main"]

START, END = "2016-09-01", "2026-09-01"  # 10 years, monthly

TICKERS_US = [
    # 21 mega-caps with full history over the window
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "BAC", "JNJ",
    "PG", "XOM", "CVX", "KO", "PEP", "WMT", "HD", "DIS", "CSCO", "INTC",
    "MRK", "VZ",
    # 9 SPDR sector ETFs with 10+ years of history (XLRE/XLC excluded: too young)
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB",
]

TICKERS_SG = [  # ~STI universe; VERIFY current membership before final use
    "D05.SI", "O39.SI", "U11.SI",            # DBS, OCBC, UOB
    "Z74.SI", "S63.SI", "C6L.SI", "S68.SI",  # Singtel, ST Eng, SIA, SGX
    "9CI.SI", "BN4.SI", "U96.SI", "F34.SI",  # CLI, Keppel, Sembcorp, Wilmar
    "G13.SI", "C09.SI", "C52.SI", "Y92.SI",  # Genting, CityDev, ComfortDelGro, ThaiBev
    "BS6.SI", "H78.SI", "J36.SI", "C07.SI",  # YZJ, HK Land, Jardine Matheson, Jardine C&C
    "D01.SI", "U14.SI", "V03.SI", "S58.SI",  # DFI Retail, UOL, Venture, SATS
    "A17U.SI", "C38U.SI", "M44U.SI",         # CLAR, CICT, MLT (REITs)
    "ME8U.SI", "N2IU.SI", "BUOU.SI", "AJBU.SI",  # MIT, MPACT, FLCT, Keppel DC
]

PRICE_UNIVERSES = {"us": TICKERS_US, "sg": TICKERS_SG}


# --------------------------------------------------------------------- prices
def fetch_yfinance(tickers: list[str]) -> pd.DataFrame:
    """One batch call for monthly adjusted closes; returns a wide DataFrame."""
    import yfinance as yf

    px = yf.download(
        tickers, start=START, end=END, interval="1mo", auto_adjust=True, progress=False
    )["Close"]
    if isinstance(px, pd.Series):  # single ticker edge case
        px = px.to_frame(tickers[0])
    return px


def fetch_stooq(ticker: str) -> pd.Series | None:
    """Fallback for one ticker via Stooq (daily -> month-end)."""
    try:
        from pandas_datareader import data as pdr

        daily = pdr.DataReader(ticker, "stooq", START, END)["Close"].sort_index()
        return daily.resample("ME").last()
    except Exception:  # noqa: BLE001 - thin coverage, treat as unresolved
        return None


def build_price_universe(tickers: list[str], name: str, report: list[str]) -> None:
    """Fetch one universe's monthly adjusted closes and write its price CSV."""
    px = pd.DataFrame()
    try:
        px = fetch_yfinance(tickers)
        report.append(f"[{name}] yfinance batch: {px.shape[1]} of {len(tickers)} tickers")
    except Exception as exc:  # noqa: BLE001 - report and try the fallback
        report.append(f"[{name}] yfinance batch FAILED: {exc!r}")

    missing = [t for t in tickers if t not in px.columns or px[t].dropna().empty]
    for ticker in missing:  # one-by-one fallback, politely paced
        series = fetch_stooq(ticker)
        if series is not None and not series.dropna().empty:
            px[ticker] = series
            report.append(f"[{name}] {ticker}: recovered via Stooq")
        else:
            report.append(f"[{name}] {ticker}: MISSING (rerun later or drop)")
        time.sleep(1)

    px = px.sort_index()
    px.to_csv(config.DATA_DIR / f"prices_{name}_monthly.csv")
    n_obs = px.notna().sum().min() if not px.empty else 0
    report.append(
        f"[{name}] saved {px.shape[1]} columns, {len(px)} rows, "
        f"thinnest ticker has {n_obs} observations\n"
    )


def fetch_ff3(report: list[str]) -> None:
    """Download the Fama–French 3 factors + RF (monthly, decimals)."""
    try:
        from pandas_datareader import data as pdr

        ff = pdr.DataReader("F-F_Research_Data_Factors", "famafrench", START, END)[0] / 100.0
        ff.to_csv(config.DATA_DIR / "ff3_monthly.csv")
        report.append(f"[ff3] saved {len(ff)} months (Mkt-RF, SMB, HML, RF)")
    except Exception as exc:  # noqa: BLE001 - manual fallback is documented
        report.append(
            f"[ff3] FAILED: {exc!r} — download the CSV manually from "
            "Ken French's data library page instead"
        )


def fetch_long_history() -> int:
    """Download Ken French's 10 industries (value-weighted) and the long factors.

    Writes the three :data:`config.LONG_HISTORY` files in decimals: industries
    from :data:`config.LONG_HISTORY_START`, the FF3 factors + RF and the
    momentum factor from :data:`config.LONG_FACTORS_START`.  Ken French marks
    missing values with -99.99; the download is refused if any appear.
    """
    from pandas_datareader import data as pdr

    files = config.LONG_HISTORY
    tables = {
        "industries": ("10_Industry_Portfolios", config.LONG_HISTORY_START),
        "ff3": ("F-F_Research_Data_Factors", config.LONG_FACTORS_START),
        "momentum": ("F-F_Momentum_Factor", config.LONG_FACTORS_START),
    }
    for key, (name, start) in tables.items():
        # Table 0 is the monthly series (value-weighted for the industries).
        frame = pdr.DataReader(name, "famafrench", start=start, end=END)[0]
        frame.columns = [c.strip() for c in frame.columns]
        if (frame <= -99.99).any().any():
            raise ValueError(f"{name}: Ken French missing-value marker (-99.99) present")
        frame = frame / 100.0
        frame.index = frame.index.strftime("%Y-%m")
        frame.index.name = "Date"
        frame.to_csv(config.DATA_DIR / files[key])
        span = f"{frame.index[0]} .. {frame.index[-1]}"
        print(f"[long-history] {files[key]}: {len(frame)} months, {span}")
    return 0


def download_prices_and_factors() -> int:
    """Fetch prices for every universe plus the FF3 factors and a coverage report."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    report: list[str] = [f"FE5108 data download, window {START} .. {END}\n"]
    for name, tickers in PRICE_UNIVERSES.items():
        build_price_universe(tickers, name, report)
    fetch_ff3(report)
    (config.DATA_DIR / "coverage_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print("\nDone. Files are in ./data — post the folder to Canvas as-is.")
    return 0


# ----------------------------------------------------------------- market caps
def read_prices(path: Path) -> pd.DataFrame:
    """Read a monthly price panel (first column is the date)."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def fetch_ticker_raw(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame | None, str]:
    """Pull prices + shares outstanding for one ticker in a single pass.

    Returns ``(frame, source)`` where ``frame`` is indexed by month-start with
    ``close`` (split-unadjusted), ``adj_close`` (total-return adjusted) and
    ``shares`` columns (forward-filled from their report dates).  ``(None,
    reason)`` is returned when nothing usable came back.
    """
    import yfinance as yf

    handle = yf.Ticker(ticker)
    history = handle.history(start=start, end=end, interval="1mo", auto_adjust=False, actions=True)
    if history.empty:
        return None, "no price history"

    index = pd.DatetimeIndex(history.index).tz_localize(None)
    frame = pd.DataFrame(index=index)
    splits = history["Stock Splits"] if "Stock Splits" in history else pd.Series(dtype=float)
    frame["close"] = unadjust_close(history["Close"], splits).to_numpy()
    frame["adj_close"] = (
        history["Adj Close"].to_numpy() if "Adj Close" in history else frame["close"]
    )

    shares = handle.get_shares_full(start=start, end=end)
    if shares is None or len(shares) == 0:
        frame["shares"] = pd.NA
        return frame, "no shares history"

    shares = shares.copy()
    shares.index = pd.DatetimeIndex(shares.index).tz_localize(None)
    shares = shares.sort_index()
    # Yahoo returns several share-count entries per date; keep the latest and
    # forward-fill onto the monthly index so a month sees only counts reported
    # on or before that month's start (no look-ahead).
    shares = shares[~shares.index.duplicated(keep="last")]
    frame["shares"] = shares.reindex(shares.index.union(frame.index)).ffill().reindex(frame.index)
    return frame, "prices + shares"


def fetch_cap_universe(name: str, filename: str, *, pace: float, refresh: bool) -> None:
    """Fetch every ticker in a universe into the raw per-ticker cache."""
    path = config.DATA_DIR / filename
    if not path.exists():
        print(f"[{name}] {path.name} not found, skipping")
        return

    prices = read_prices(path)
    raw_dir = config.raw_universe_dir(name)
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = prices.index.min() - pd.Timedelta(days=400)
    end = prices.index.max() + pd.Timedelta(days=1)
    print(f"[{name}] {prices.shape[1]} tickers, {start:%Y-%m-%d} .. {end:%Y-%m-%d} -> {raw_dir}")

    for ticker in prices.columns:
        out_path = raw_dir / f"{ticker}.csv"
        if out_path.exists() and not refresh:
            print(f"  {ticker:9s} cached")
            continue
        try:
            frame, source = fetch_ticker_raw(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            frame, source = None, f"error: {exc!r}"

        if frame is None:
            print(f"  {ticker:9s} MISSING  ({source})")
        else:
            frame.to_csv(out_path)
            n_shares = int(frame["shares"].notna().sum())
            print(
                f"  {ticker:9s} {len(frame):3d} months, "
                f"{n_shares:3d} share observations  ({source})"
            )
        time.sleep(pace)


def build_cap_universe(name: str) -> None:
    """Convert the raw cache for a universe into its long-format weight CSV.

    Singapore weights come from :func:`build_sg` instead: Yahoo's scaled
    close and glitched share counts make the plain raw-cache caps wrong there.
    """
    if name == "sg":
        build_sg()
        return
    path = config.DATA_DIR / f"prices_{name}_monthly.csv"
    assets = read_prices(path).columns if path.exists() else None
    long = build_value_weights_from_raw(name, assets=assets)

    out = config.DATA_DIR / f"value_weights_{name}.csv"
    long.to_csv(out, index=False)
    n_assets = long["asset"].nunique()
    n_dates = long["date"].nunique()
    print(f"[{name}] wrote {out.name}: {n_assets} assets x {n_dates} months")


def fetch_value_weights(
    *,
    stage: str = "all",
    universe: str | None = None,
    pace: float = 0.5,
    refresh: bool = False,
) -> int:
    """Run the market-cap stages (raw cache fetch and/or offline weight build)."""
    files = {name: f"prices_{name}_monthly.csv" for name in PRICE_UNIVERSES}
    names = {universe: files[universe]} if universe else files
    if stage in {"all", "fetch"}:
        for name, filename in names.items():
            fetch_cap_universe(name, filename, pace=pace, refresh=refresh)
    if stage in {"all", "build"}:
        for name in names:
            build_cap_universe(name)
    return 0


# ------------------------------------------------------------ SG daily cache
SGX_HISTORY_URL = "https://api.sgx.com/securities/v1.1/charts/historic/{kind}/code/{code}/all"
SGX_KINDS = ("stocks", "reits")  # REITs are served under their own path
FX_TICKER = "SGD=X"  # Yahoo's USD/SGD rate: SGD per USD


def fetch_sgx_history(code: str) -> pd.DataFrame:
    """SGX's own daily last-traded price for one stock code, never adjusted.

    The SGX website's historical-price service returns every session since
    2011, under ``stocks`` for companies and ``reits`` for REITs (the
    ``stocks`` path holds only a fragment of a REIT's history), so the longer
    of the two is kept.  Some days appear more than once (one row per
    snapshot); the latest snapshot of each day is kept.
    """
    import requests  # installed with yfinance (the ``data`` extra)

    histories = []
    for kind in SGX_KINDS:
        response = requests.get(
            SGX_HISTORY_URL.format(kind=kind, code=code),
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=60,
        )
        response.raise_for_status()
        histories.append(response.json()["data"]["historic"])
    frame = pd.DataFrame(max(histories, key=len))
    frame["date"] = pd.to_datetime(frame["trading_time"].str[:8])
    frame = frame.sort_values("trading_time").groupby("date").last()
    return pd.DataFrame(
        {"sgx_close": frame["lt"].astype(float), "currency": frame["cur"]},
        index=frame.index,
    )


def fetch_sg_daily_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Yahoo's daily close and dividends side by side with SGX's raw price.

    ``close`` is Yahoo's ``Close`` with ``auto_adjust=False``: split- and
    capital-adjusted, never dividend-adjusted.  ``dividend`` and ``split`` are
    Yahoo's corporate-action columns and ``adj_close`` its total-return series
    (kept only so the cleaning report can show what changed).  ``sgx_close``
    is the exchange's own unadjusted last-traded price for the same day.
    """
    import yfinance as yf

    history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=True)
    if history.empty:
        raise ValueError(f"No Yahoo history for {ticker}")
    index = pd.DatetimeIndex(history.index).tz_localize(None).normalize()
    frame = pd.DataFrame(
        {
            "close": history["Close"].to_numpy(),
            "adj_close": history["Adj Close"].to_numpy(),
            "dividend": history["Dividends"].to_numpy(),
            "split": history["Stock Splits"].to_numpy(),
        },
        index=index,
    )
    sgx = fetch_sgx_history(ticker.removesuffix(".SI"))
    frame = frame.join(sgx, how="left")
    frame.index.name = "date"
    return frame


def fetch_fx(start: str, end: str) -> pd.Series:
    """Month-end USD/SGD (SGD per USD), labelled by month start like the prices."""
    import yfinance as yf

    history = yf.Ticker(FX_TICKER).history(start=start, end=end, interval="1mo")
    index = pd.DatetimeIndex(history.index).tz_localize(None).normalize()
    return pd.Series(history["Close"].to_numpy(), index=index, name="usdsgd")


def fetch_sg_daily(*, pace: float = 0.5, refresh: bool = False) -> int:
    """Fill the SG daily raw cache (Yahoo + SGX) and the USD/SGD series."""
    out_dir = config.RAW_DIR / "sg_daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = (pd.Timestamp(START) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    for ticker in TICKERS_SG:
        path = out_dir / f"{ticker}.csv"
        if path.exists() and not refresh:
            print(f"  {ticker:9s} cached")
            continue
        try:
            frame = fetch_sg_daily_ticker(ticker, start, END)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  {ticker:9s} FAILED ({exc!r})")
            continue
        frame.to_csv(path)
        n_sgx = int(frame["sgx_close"].notna().sum())
        print(f"  {ticker:9s} {len(frame):5d} days, {n_sgx:5d} with an SGX price")
        time.sleep(pace)

    fx_dir = config.RAW_DIR / "fx"
    fx_dir.mkdir(parents=True, exist_ok=True)
    fx = fetch_fx(start, END)
    fx.rename_axis("date").to_csv(fx_dir / "usdsgd_monthly.csv")
    print(f"[fx] {len(fx)} months of USD/SGD -> {fx_dir / 'usdsgd_monthly.csv'}")
    return 0


def build_sg() -> int:
    """Rebuild the clean SG prices, value weights and cleaning report (offline)."""
    result = build_sg_clean()
    result.prices.to_csv(config.DATA_DIR / "prices_sg_monthly.csv")
    result.value_weights.to_csv(config.DATA_DIR / "value_weights_sg.csv", index=False)
    (config.DATA_DIR / "cleaning_report_sg.txt").write_text(result.report)
    print(result.report)
    return 0


# ------------------------------------------------------------------- CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goldman.data.fetch", description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=[
            "all", "prices", "caps", "caps-fetch", "caps-build", "sg-daily", "sg-build",
            "long-history",
        ],
        default="all",
        help=(
            "What to fetch: prices (prices + FF3), caps (both market-cap stages), "
            "caps-fetch (network -> raw cache), caps-build (offline), "
            "sg-daily (network -> SG daily cache), sg-build (offline), "
            "long-history (Ken French industries + factors since 1926), all (default)."
        ),
    )
    parser.add_argument(
        "--pace", type=float, default=0.5, help="Seconds between tickers (default: 0.5)."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch tickers even when a raw cache file already exists.",
    )
    parser.add_argument(
        "--universe",
        choices=sorted(PRICE_UNIVERSES),
        default=None,
        help="Limit market-cap stages to one universe (default: all).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.stage in {"all", "prices"}:
        download_prices_and_factors()
    if args.stage in {"all", "long-history"}:
        fetch_long_history()
    if args.stage == "all":
        # The SG daily cache must exist before the cap build, which for SG
        # rebuilds the clean prices and weights from it.
        fetch_sg_daily(pace=args.pace, refresh=args.refresh)
        return fetch_value_weights(stage="all", pace=args.pace, refresh=args.refresh)
    if args.stage == "sg-daily":
        return fetch_sg_daily(pace=args.pace, refresh=args.refresh)
    if args.stage == "sg-build":
        return build_sg()
    if args.stage == "caps":
        return fetch_value_weights(
            stage="all", universe=args.universe, pace=args.pace, refresh=args.refresh
        )
    if args.stage == "caps-fetch":
        return fetch_value_weights(
            stage="fetch", universe=args.universe, pace=args.pace, refresh=args.refresh
        )
    if args.stage == "caps-build":
        return fetch_value_weights(stage="build", universe=args.universe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())