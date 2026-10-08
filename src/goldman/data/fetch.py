"""Explicit, network-only fetchers — never imported by the analysis pipeline.

Everything that touches the internet lives here and runs **only when invoked**;
importing ``goldman.data`` (or running the notebook) never calls it.
The optional ``data`` extra (yfinance, pandas-datareader) is imported lazily
inside each function, so the rest of the package works without it.

Usage::

    uv sync --extra data
    uv run python -m goldman.data.fetch prices       # prices + Ken French data
    uv run python -m goldman.data.fetch factors      # Ken French factors + portfolios only
    uv run python -m goldman.data.fetch proxies      # S&P 500 + MSCI World ETFs (Stage 4)
    uv run python -m goldman.data.fetch caps         # fetch + build value weights
    uv run python -m goldman.data.fetch caps-fetch   # network -> raw per-ticker cache
    uv run python -m goldman.data.fetch caps-build   # raw cache -> value weights (offline)
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
* Prices are Yahoo's adjusted close (splits + dividends) — a total-return proxy.
* The SG list is "approximately the STI": verify current membership before
  relying on the value-weight comparison.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .. import config
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


# One Ken French dataset per cached CSV: (library name, table index, start,
# values in percent?).  Table 0 is the monthly series; for the portfolio files
# it is the value-weighted one.  Firm counts and sizes are not percentages.
_INDUSTRIES = "10_Industry_Portfolios"  # table 4: number of firms, 5: average size ($m)
KEN_FRENCH_DATASETS: dict[str, tuple[str, int, str, bool]] = {
    config.FF3_FILENAME: ("F-F_Research_Data_Factors", 0, START, True),
    config.FF5_FILENAME: ("F-F_Research_Data_5_Factors_2x3", 0, START, True),
    config.MOMENTUM_FILENAME: ("F-F_Momentum_Factor", 0, START, True),
    config.TEST_PORTFOLIOS["ff25"]: ("25_Portfolios_5x5", 0, START, True),
    config.TEST_PORTFOLIOS["mom10"]: ("10_Portfolios_Prior_12_2", 0, START, True),
    config.DEVELOPED_FILENAME: ("Developed_3_Factors", 0, START, True),
    config.LONG_HISTORY["returns"]: (_INDUSTRIES, 0, config.LONG_HISTORY_START, True),
    config.LONG_HISTORY["n_firms"]: (_INDUSTRIES, 4, config.LONG_HISTORY_START, False),
    config.LONG_HISTORY["avg_size"]: (_INDUSTRIES, 5, config.LONG_HISTORY_START, False),
    config.LONG_HISTORY["ff3"]: ("F-F_Research_Data_Factors", 0, config.LONG_FACTORS_START, True),
    config.LONG_HISTORY["momentum"]: ("F-F_Momentum_Factor", 0, config.LONG_FACTORS_START, True),
}


def fetch_ken_french(report: list[str]) -> None:
    """Download every Ken French file in one pass (monthly, decimals).

    All files come from the same library vintage, so factors and test
    portfolios stay consistent with each other.  Ken French marks missing
    values with -99.99 / -999; those become NaN before rescaling.
    """
    try:
        from pandas_datareader import data as pdr
    except Exception as exc:  # noqa: BLE001 - optional extra not installed
        report.append(f"[ken-french] FAILED: {exc!r} (run `uv sync --extra data`)")
        return

    cache: dict[tuple[str, str], dict] = {}
    for filename, (dataset, table, start, percent) in KEN_FRENCH_DATASETS.items():
        try:
            if (dataset, start) not in cache:
                cache[(dataset, start)] = pdr.DataReader(dataset, "famafrench", start, END)
            frame = cache[(dataset, start)][table]
        except Exception as exc:  # noqa: BLE001 - manual fallback is documented
            report.append(
                f"[{dataset}] FAILED: {exc!r} — download the CSV manually from "
                "Ken French's data library page instead"
            )
            continue
        frame = frame.rename(columns=str.strip).replace([-99.99, -999.0], float("nan"))
        if percent:
            frame = frame / 100.0
        frame.index.name = "Date"
        frame.to_csv(config.DATA_DIR / filename)
        report.append(
            f"[{dataset} table {table}] saved {len(frame)} months x {frame.shape[1]} columns "
            f"({frame.index[0]} .. {frame.index[-1]}) -> {filename}"
        )


def download_index_proxies() -> int:
    """Fetch the S&P 500 and MSCI World ETFs (monthly adjusted closes)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickers = list(config.INDEX_PROXIES)
    px = fetch_yfinance(tickers).sort_index()
    px = px[[t for t in tickers if t in px.columns]]
    px.to_csv(config.DATA_DIR / config.INDEX_PROXIES_FILENAME)
    print(
        f"[proxies] saved {px.shape[1]} of {len(tickers)} tickers, {len(px)} months "
        f"-> {config.INDEX_PROXIES_FILENAME}"
    )
    return 0


def download_factors() -> int:
    """Fetch only the Ken French factors and test portfolios (no prices)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    report: list[str] = [f"FE5108 Ken French download, window {START} .. {END}\n"]
    fetch_ken_french(report)
    (config.DATA_DIR / "coverage_factors.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


def download_prices_and_factors() -> int:
    """Fetch prices for every universe plus the Ken French data and a coverage report."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    report: list[str] = [f"FE5108 data download, window {START} .. {END}\n"]
    for name, tickers in PRICE_UNIVERSES.items():
        build_price_universe(tickers, name, report)
    fetch_ken_french(report)
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
    """Convert the raw cache for a universe into its long-format weight CSV."""
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


# ------------------------------------------------------------------- CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goldman.data.fetch", description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=["all", "prices", "factors", "proxies", "caps", "caps-fetch", "caps-build"],
        default="all",
        help=(
            "What to fetch: prices (prices + Ken French), factors (Ken French only), "
            "proxies (S&P 500 and MSCI World ETFs), "
            "caps (both market-cap stages), "
            "caps-fetch (network -> raw cache), caps-build (offline), all (default)."
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
    if args.stage in {"all", "proxies"}:
        download_index_proxies()
    if args.stage == "proxies":
        return 0
    if args.stage == "factors":
        return download_factors()
    if args.stage == "all":
        return fetch_value_weights(stage="all", pace=args.pace, refresh=args.refresh)
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