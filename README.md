# FE 5108 Midterm Project — Fallback Data Bundle

This folder is the safety net promised in the handout's *Data and tools*
section: if Yahoo rate-limits your group, you can start (and finish) Stage 1
from here instead of stalling.

## Contents

| File | What it is |
|---|---|
| `download_data.py` | One-shot downloader: 30 US names (21 mega-caps + 9 SPDR sector ETFs), ~30 large SGX names, and the Fama–French 3 factors + RF. Run it **once** on an open network; it writes clean monthly CSVs into `data/` plus a `coverage_report.txt`. Falls back to Stooq per ticker if Yahoo throttles. |
| `starter_stage1.py` / `starter_stage1.ipynb` | Stage 1 skeleton (same code, script and notebook). Reads a price CSV, produces exhibits 1a/1b/1c. Needs only numpy, pandas, matplotlib — no internet. |
| `data/sample_synthetic_prices.csv` | **Fake** data (30 assets × 120 months, labelled SYN01–SYN30) so the starter runs before you have real data. Do not use it in your report. |

## For the instructor (before posting to Canvas)

1. `pip install yfinance pandas-datareader`
2. `python download_data.py` — takes a minute; read `coverage_report.txt`.
3. Post the whole folder (now including the real CSVs in `data/`) to Canvas.

## Notes and honest caveats

- Prices are Yahoo **adjusted closes** (splits + dividends) — a total-return
  proxy. Say so in your report.
- The SG list is *approximately* the STI. Index membership changes; verify
  against the current constituents before using value weights.
- Both universes are **today's members backfilled ten years** — the
  survivorship bias the handout tells you to acknowledge and discuss.
- The starter compares tangency weights with **1/N**; the handout asks for
  **value weights** — get market caps and swap them in.
- Sector ETFs XLRE and XLC are excluded on purpose (listed 2015/2018 — not
  enough history for the 96-month minimum).

## What the starter will show you (yes, it's supposed to look like that)

On the synthetic sample it already produces a −73% short position in the
tangency portfolio and a correlation of **0.16** between weight vectors
estimated on the two halves of the sample. That is estimation error, the
central character of Stage 1 — report it, don't "fix" it.
