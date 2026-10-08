"""Analysis: portfolio construction and the report exhibits.

* :mod:`.portfolio_choice` — Stage 1, mean–variance choice and estimation error;
* :mod:`.capm` — Stage 2, the CAPM regressions and the empirical SML;
* :mod:`.factor_models` — Stage 3, multi-factor alphas;
* :mod:`.roll` and :mod:`.backtest` — Stage 4, Roll's critique and the
  out-of-sample horse race behind the recommendation;
* :mod:`.regression` — the shared econometrics (OLS, GRS, Fama–MacBeth);
* :mod:`.estimators` — Ledoit–Wolf covariance and CAPM-prior mean shrinkage.
"""

from __future__ import annotations

from . import (
    backtest,
    capm,
    estimators,
    factor_models,
    portfolio,
    portfolio_choice,
    regression,
    roll,
    self_built,
)
from .backtest import (
    STANDARD_RULES,
    BacktestResult,
    historical_premium,
    holding_weights,
    performance_table,
    plot_cumulative,
    rolling_backtest,
    sharpe_difference_test,
    tangency_sign_flips,
)
from .capm import (
    CapmAnalysisResult,
    SMLResult,
    alpha_joint_test,
    capm_regressions,
    capm_table,
    plot_sml,
    run_capm_analysis,
    security_market_line,
    sml_comparison,
)
from .estimators import (
    capm_prior_means,
    ledoit_wolf_covariance,
    shrink_means,
    shrinkage_intensity,
)
from .factor_models import (
    MODEL_FACTORS,
    FactorAnalysisResult,
    alpha_comparison,
    anomaly_verdict,
    factor_loadings,
    factor_summary,
    fit_models,
    model_summary,
    plot_alpha_comparison,
    run_factor_analysis,
    standard_models,
)
from .portfolio import (
    equal_weights,
    excess_means,
    excess_returns,
    min_variance_weights,
    min_variance_weights_long_only,
    normalize_weights,
    subsample_tangency,
    tangency_weights,
    tangency_weights_long_only,
    two_fund_allocation,
    weight_correlation,
)
from .portfolio_choice import (
    PortfolioAnalysisResult,
    TamingResult,
    plot_taming,
    plot_weights,
    run_portfolio_analysis,
    subset_tangency_check,
    summary_statistics,
    taming_comparison,
    tangency_vs_comparison,
)
from .regression import (
    FamaMacBethResult,
    GRSResult,
    TimeSeriesFit,
    chance_exceedances,
    fama_macbeth,
    grs_test,
    newey_west_lags,
    time_series_regression,
)
from .roll import (
    RollResult,
    ex_post_tangency_proxy,
    max_sharpe,
    proxy_comparison,
    run_roll_analysis,
)
from .self_built import (
    SelfBuiltFactor,
    factor_report,
    long_short_factor,
    momentum_signal,
)

__all__ = [
    "backtest",
    "capm",
    "estimators",
    "factor_models",
    "portfolio",
    "portfolio_choice",
    "regression",
    "roll",
    "self_built",
    # Stage 1
    "PortfolioAnalysisResult",
    "run_portfolio_analysis",
    "summary_statistics",
    "tangency_vs_comparison",
    "plot_weights",
    "equal_weights",
    "excess_means",
    "excess_returns",
    "normalize_weights",
    "subsample_tangency",
    "tangency_weights",
    "tangency_weights_long_only",
    "min_variance_weights",
    "min_variance_weights_long_only",
    "weight_correlation",
    "TamingResult",
    "taming_comparison",
    "subset_tangency_check",
    "two_fund_allocation",
    "plot_taming",
    # Stage 2
    "CapmAnalysisResult",
    "SMLResult",
    "alpha_joint_test",
    "capm_regressions",
    "capm_table",
    "plot_sml",
    "run_capm_analysis",
    "security_market_line",
    "sml_comparison",
    # Stage 3
    "MODEL_FACTORS",
    "FactorAnalysisResult",
    "alpha_comparison",
    "anomaly_verdict",
    "factor_loadings",
    "factor_summary",
    "fit_models",
    "model_summary",
    "plot_alpha_comparison",
    "run_factor_analysis",
    "standard_models",
    "SelfBuiltFactor",
    "factor_report",
    "long_short_factor",
    "momentum_signal",
    # Stage 4
    "RollResult",
    "ex_post_tangency_proxy",
    "max_sharpe",
    "proxy_comparison",
    "run_roll_analysis",
    "STANDARD_RULES",
    "BacktestResult",
    "historical_premium",
    "holding_weights",
    "performance_table",
    "plot_cumulative",
    "rolling_backtest",
    "sharpe_difference_test",
    "tangency_sign_flips",
    # estimators
    "capm_prior_means",
    "ledoit_wolf_covariance",
    "shrink_means",
    "shrinkage_intensity",
    # shared econometrics
    "FamaMacBethResult",
    "GRSResult",
    "TimeSeriesFit",
    "chance_exceedances",
    "fama_macbeth",
    "grs_test",
    "newey_west_lags",
    "time_series_regression",
]
