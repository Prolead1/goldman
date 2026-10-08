"""Reproduce alternative Stage-3 exhibits for US and Singapore."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from goldman.analysis.alternative_factors import alternative_factors
from goldman.analysis.factor_models import run_factor_analysis
from goldman.data import risk_free_series
from goldman.data.factors import load_ff3
from goldman.data.markets import market_excess_returns
from goldman.data.prices import load_returns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "alternative_factors"
OUT.mkdir(parents=True, exist_ok=True)

for universe in ("us", "sg"):
    returns = load_returns(universe=universe).dropna()
    rf = risk_free_series(universe, returns.index)
    market = market_excess_returns(returns, universe=universe)
    if isinstance(market, pd.DataFrame):
        market = market.iloc[:, 0]
    market = market.rename("MKT")
    extra = alternative_factors(returns)
    factors = pd.concat([market, extra], axis=1)
    models = {"CAPM": factors[["MKT"]], "Market LOWVOL REV": factors}
    if universe == "us":
        ff = load_ff3()
        ff.index = ff.index.to_timestamp()
        models["Market HML"] = pd.concat([market, ff[["HML"]]], axis=1)
    result = run_factor_analysis(
        returns, universe=universe, models=models, rf=rf,
        output_dir=OUT / universe,
    )
    extra.to_csv(OUT / f"{universe}_factor_returns.csv")
    result.alphas.add_prefix("alpha_").to_csv(OUT / f"{universe}_alpha_comparison.csv")
    result.summary.to_csv(OUT / f"{universe}_model_summary.csv")
    (1 + extra.dropna()).cumprod().plot(figsize=(9, 5), title=f"{universe.upper()}: alternative factors")
    plt.ylabel("Growth of $1 (gross, long-short)")
    plt.tight_layout()
    plt.savefig(OUT / f"{universe}_factor_growth.png", dpi=160)
    plt.close()
    print(universe, len(result.sample), "months", result.summary.to_string())
