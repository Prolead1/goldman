# Stage 3 extension: low volatility and short-term reversal

This additive experiment is **not** the existing SMB/WML implementation. It runs on
both US and Singapore price universes and uses the repository's existing
`run_factor_analysis` to compare CAPM and a market + LOWVOL + REV model on an
identical estimation window. It does not overwrite existing Stage 1/2 exhibits.

**Low volatility (LOWVOL).** For each holding month t, compute each stock's
sample standard deviation over the previous 12 monthly returns (t-12 through
t-1). Sort eligible stocks into terciles, buy the lowest-volatility tercile
and short the highest-volatility tercile, equally weighting each leg. The
spread is the current month's long return minus short return. This probes
whether low-risk stocks earn abnormal returns relative to their market betas.
It is a volatility-sort factor, **not** a direct test of a profitability or
book-to-market characteristic.

**Short-term reversal (REV).** At each t, rank stocks on their return in t-1,
long the bottom tercile (last month's losers), and short the top tercile
(last month's winners), equally weighted. The current-month spread tests
whether short-run price reversal explains residual CAPM alphas. This is
**distinct** from the project's 12-2 momentum factor.

## Run

```bash
uv sync --extra notebook
uv run python scripts/run_alternative_factors.py
uv run pytest
uv run ruff check .
```

Outputs go to `outputs/alternative_factors/`, which is git-ignored. Compare
`alpha_CAPM` and `alpha_Market_LOWVOL_REV` in each `*_alpha_comparison.csv`;
check t-statistics and GRS p-values in the model tables/summary before making
claims about anomalies. Regressions use **in-sample factor returns constructed
from the tested assets**; any alpha reduction is descriptive, not independent
out-of-sample proof of a risk premium. The bundled universe mixes individual
stocks and ETFs; interpretation should acknowledge this. The strategy assumes
frictionless shorting, equal weighting, and no transaction costs.

For the US only, Ken French's HML is an optional independent benchmark; it is
not a self-built SG value factor because book-to-market fundamentals are absent.
