"""
FE 5108 Midterm Project — Stage 1 starter (script version; a notebook version
starter_stage1.ipynb with the same cells sits next to this file).

Reads a monthly price CSV from ./data/ and produces Stage 1's required
exhibits: (1a) summary statistics, (1b) tangency weights next to naive
weights as one chart, (1c) the subsample re-estimation that shows the
weights jumping — i.e. estimation error made visible.

Out of the box it runs on data/sample_synthetic_prices.csv (fake data,
shipped so you can test your environment).  Point PRICES_CSV at
prices_us_monthly.csv or prices_sg_monthly.csv once you have real data.

Only numpy / pandas / matplotlib are needed.  No internet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- settings
PRICES_CSV = Path(__file__).parent / "data" / "sample_synthetic_prices.csv"
RF_MONTHLY = 0.003            # replace with RF from ff3_monthly.csv / MAS
MIN_OBS = 90                  # drop assets with fewer clean observations

# ------------------------------------------------------------- load & clean
px = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).sort_index()
px = px.dropna(axis=1, thresh=MIN_OBS)          # kill sparse tickers
rets = px.pct_change().dropna(how="all")        # simple monthly returns
rets = rets.dropna(axis=0)                      # keep fully-observed months
print(f"Universe: {rets.shape[1]} assets, {rets.shape[0]} months "
      f"({rets.index[0]:%Y-%m} .. {rets.index[-1]:%Y-%m})")

# ------------------------------------------------- exhibit 1a: summary stats
summary = pd.DataFrame({
    "mean (ann.)":  rets.mean() * 12,
    "vol (ann.)":   rets.std() * np.sqrt(12),
    "sharpe (ann.)": (rets.mean() - RF_MONTHLY) / rets.std() * np.sqrt(12),
    "min month":    rets.min(),
    "max month":    rets.max(),
}).round(3)
summary.to_csv("exhibit_1a_summary_stats.csv")
print("\nExhibit 1a written: exhibit_1a_summary_stats.csv")

# --------------------------------------- exhibit 1b: tangency vs 1/N weights
mu_e = rets.mean().values - RF_MONTHLY          # monthly excess means
Sigma = rets.cov().values
w_tan = np.linalg.solve(Sigma, mu_e)
w_tan = w_tan / w_tan.sum()                     # normalise to sum to 1
w_naive = np.full(len(mu_e), 1 / len(mu_e))
# NOTE: the handout asks for VALUE weights as the comparison.  Get market
# caps (shares outstanding x price) for your universe and replace w_naive.

fig, ax = plt.subplots(figsize=(12, 4))
ix = np.arange(len(mu_e))
ax.bar(ix - 0.2, w_tan, 0.4, label="tangency  $\\hat\\Sigma^{-1}\\hat\\mu_e$")
ax.bar(ix + 0.2, w_naive, 0.4, label="1/N (replace with value weights)")
ax.set_xticks(ix); ax.set_xticklabels(rets.columns, rotation=90, fontsize=7)
ax.axhline(0, lw=0.8, color="k"); ax.legend()
ax.set_title("Exhibit 1b — estimated tangency weights vs naive weights")
fig.tight_layout(); fig.savefig("exhibit_1b_weights.png", dpi=150)
print(f"Exhibit 1b written: exhibit_1b_weights.png "
      f"(most extreme tangency position: {w_tan[np.abs(w_tan).argmax()]:+.0%})")

# ----------------------- exhibit 1c: re-estimate on halves, watch weights move
half = len(rets) // 2
out = {}
for label, sub in [("first half", rets.iloc[:half]), ("second half", rets.iloc[half:])]:
    m = sub.mean().values - RF_MONTHLY
    w = np.linalg.solve(sub.cov().values, m)
    out[label] = w / w.sum()
drift = pd.DataFrame(out, index=rets.columns)
drift["full sample"] = w_tan
drift.round(3).to_csv("exhibit_1c_subsample_weights.csv")
corr = drift["first half"].corr(drift["second half"])
print(f"Exhibit 1c written: exhibit_1c_subsample_weights.csv "
      f"(corr between half-sample weight vectors: {corr:.2f} — "
      "if this is far below 1, you are looking at estimation error)")

print("\nStage 1 skeleton complete. Your job: value weights, discussion, "
      "and the honest sentence about what 1c implies for trusting 1b.")
