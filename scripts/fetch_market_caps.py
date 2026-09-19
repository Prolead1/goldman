"""Fetch current market caps via yfinance and write value-weight CSVs.

For every price panel in ``data/`` (US and SG universes) this looks up each
ticker's market capitalisation with yfinance and writes
``data/value_weights_<name>.csv`` with columns ``asset, market_cap, weight``.
Weights are ``market_cap / sum(market_cap)``.

Run once on a machine with open internet::

    uv sync --extra data
    uv run python scripts/fetch_market_caps.py

Caveats
-------
* Market caps are a *point-in-time* (today's) snapshot, so the value-weighted
  comparison inherits the same survivorship / look-ahead bias the handout asks
  you to acknowledge.
* Tickers whose cap cannot be resolved are recorded in the console report and
  omitted; rerun later or add them manually.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

UNIVERSES = {
    "us": "prices_us_monthly.csv",
    "sg": "prices_sg_monthly.csv",
}


def read_tickers(path: Path) -> list[str]:
    """Read the ticker list from a price CSV header (first column is the date)."""
    return list(pd.read_csv(path, nrows=0).columns[1:])


def resolve_market_cap(ticker: str, last_price: float | None) -> tuple[float | None, str]:
    """Return ``(market_cap, source)`` for one ticker, falling back to shares."""
    import yfinance as yf

    info = yf.Ticker(ticker).info
    cap = info.get("marketCap")
    if cap:
        return float(cap), "marketCap"

    shares = info.get("sharesOutstanding")
    if shares and last_price:
        return float(shares) * float(last_price), "shares x last price"
    return None, "unresolved"


def fetch_market_caps(prices: pd.DataFrame, pace: float) -> pd.DataFrame:
    """Resolve a market cap for every ticker in ``prices``, paced politely."""
    rows = []
    for ticker in prices.columns:
        last = prices[ticker].dropna()
        last_price = float(last.iloc[-1]) if not last.empty else None
        try:
            cap, source = resolve_market_cap(ticker, last_price)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            cap, source = None, f"error: {exc!r}"
        rows.append({"asset": ticker, "market_cap": cap, "source": source})
        status = f"{cap:,.0f}" if cap else "MISSING"
        print(f"  {ticker:9s} {status:>18s}  ({source})")
        time.sleep(pace)
    return pd.DataFrame(rows)


def build_value_weights(name: str, filename: str, pace: float) -> None:
    path = DATA / filename
    if not path.exists():
        print(f"[{name}] {path.name} not found, skipping")
        return

    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"[{name}] fetching market caps for {prices.shape[1]} tickers...")
    caps = fetch_market_caps(prices, pace)

    resolved = caps.dropna(subset=["market_cap"]).copy()
    unresolved = caps[caps["market_cap"].isna()]["asset"].tolist()
    if resolved.empty:
        print(f"[{name}] no caps resolved; nothing written")
        return

    resolved["weight"] = resolved["market_cap"] / resolved["market_cap"].sum()
    out = DATA / f"value_weights_{name}.csv"
    resolved[["asset", "market_cap", "weight"]].to_csv(out, index=False)

    note = f"; unresolved: {', '.join(unresolved)}" if unresolved else ""
    print(f"[{name}] wrote {out.name}: {len(resolved)} assets{note}\n")


def main() -> int:
    parser = argparse.ArgumentParser(prog="fetch_market_caps", description=__doc__)
    parser.add_argument(
        "--pace",
        type=float,
        default=0.5,
        help="Seconds to sleep between tickers (default: 0.5).",
    )
    args = parser.parse_args()

    for name, filename in UNIVERSES.items():
        build_value_weights(name, filename, args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())