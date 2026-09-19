"""
FE 5108 Project — fallback data downloader.

Run ONCE on a machine with open internet (instructor or student):

    pip install yfinance pandas-datareader
    python scripts/download_data.py

Produces, in ./data/ :
    prices_us_monthly.csv   30 liquid US names (21 mega-caps + 9 SPDR sector ETFs)
    prices_sg_monthly.csv   30 large SGX names (approximately the STI universe)
    ff3_monthly.csv         Fama-French 3 factors + RF (from Ken French's library)
    coverage_report.txt     what was fetched, what is missing

Design notes
------------
* ONE batch call per universe, then cache — repeated pulls are what triggers
  Yahoo's rate limiting (HTTP 429).
* If yfinance fails for a ticker, the script retries it via Stooq
  (pandas-datareader).  Stooq's Singapore coverage is thin, so SG names may
  stay missing there — rerun later, or drop them.
* Prices are Yahoo's adjusted close (splits + dividends), i.e. a total-return
  proxy — say so in the report.
* The SG list is "approximately the STI": index membership changes, so verify
  against the current STI constituents and adjust TICKERS_SG before relying
  the value-weight comparison.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

START, END = "2016-09-01", "2026-09-01"   # 10 years, monthly
OUT = Path(__file__).resolve().parents[1] / "data"   # scripts/ -> repo root/data
OUT.mkdir(exist_ok=True)

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


def fetch_yfinance(tickers: list[str]) -> pd.DataFrame:
    """One batch call for monthly adjusted closes; returns wide DataFrame."""
    import yfinance as yf
    px = yf.download(tickers, start=START, end=END, interval="1mo",
                     auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):          # single ticker edge case
        px = px.to_frame(tickers[0])
    return px


def fetch_stooq(ticker: str) -> pd.Series | None:
    """Fallback for one ticker via Stooq (daily -> month-end)."""
    try:
        from pandas_datareader import data as pdr
        d = pdr.DataReader(ticker, "stooq", START, END)["Close"].sort_index()
        return d.resample("ME").last()
    except Exception:
        return None


def build_universe(tickers: list[str], name: str, report: list[str]) -> None:
    px = pd.DataFrame()
    try:
        px = fetch_yfinance(tickers)
        report.append(f"[{name}] yfinance batch: {px.shape[1]} of {len(tickers)} tickers")
    except Exception as e:
        report.append(f"[{name}] yfinance batch FAILED: {e!r}")

    missing = [t for t in tickers if t not in px.columns or px[t].dropna().empty]
    for t in missing:                      # one-by-one fallback, politely paced
        s = fetch_stooq(t)
        if s is not None and not s.dropna().empty:
            px[t] = s
            report.append(f"[{name}] {t}: recovered via Stooq")
        else:
            report.append(f"[{name}] {t}: MISSING (rerun later or drop)")
        time.sleep(1)

    px = px.sort_index()
    px.to_csv(OUT / f"prices_{name}_monthly.csv")
    n_obs = px.notna().sum().min() if not px.empty else 0
    report.append(f"[{name}] saved {px.shape[1]} columns, "
                  f"{len(px)} rows, thinnest ticker has {n_obs} observations\n")


def fetch_ff3(report: list[str]) -> None:
    try:
        from pandas_datareader import data as pdr
        ff = pdr.DataReader("F-F_Research_Data_Factors", "famafrench",
                            START, END)[0] / 100.0   # decimals, monthly
        ff.to_csv(OUT / "ff3_monthly.csv")
        report.append(f"[ff3] saved {len(ff)} months (Mkt-RF, SMB, HML, RF)")
    except Exception as e:
        report.append(f"[ff3] FAILED: {e!r} — download the CSV manually from "
                      "Ken French's data library page instead")


if __name__ == "__main__":
    report: list[str] = [f"FE5108 data download, window {START} .. {END}\n"]
    build_universe(TICKERS_US, "us", report)
    build_universe(TICKERS_SG, "sg", report)
    fetch_ff3(report)
    (OUT / "coverage_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print("\nDone. Files are in ./data — post the folder to Canvas as-is.")
