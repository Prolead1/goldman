"""Analysis: portfolio construction and the report exhibits.

Currently the mean–variance portfolio choice analysis; further analyses (the
CAPM/SML, factor models, the decision memo) slot in here beside it.
"""

from __future__ import annotations

from . import (
    capm,
    custom_factors,
    factor_construction,
    factor_models,
    out_of_sample,
    portfolio,
    portfolio_choice,
    robustness,
)
from .capm import (
    CapmResult,
    SMLFit,
    empirical_sml,
    excess_returns,
    grs_test,
    plot_sml,
    run_capm_analysis,
    security_characteristic_line,
)
from .custom_factors import (
    build_custom_factors,
    factor_scorecard,
    near_high_signal,
    spanning_regression,
    volatility_scale,
)
from .factor_construction import (
    build_self_factors,
    long_short_spread,
    long_short_weights,
    momentum_factor,
    momentum_signal,
    size_factor,
)
from .factor_models import (
    FactorModelResult,
    alpha_comparison,
    factor_regression,
    model_summary,
    run_factor_analysis,
)
from .out_of_sample import (
    OOS_BENCHMARKS,
    OutOfSampleResult,
    out_of_sample_summary,
    run_out_of_sample_analysis,
    walk_forward_returns,
)
from .portfolio import (
    equal_weights,
    excess_means,
    normalize_weights,
    subsample_tangency,
    tangency_weights,
    weight_correlation,
)
from .portfolio_choice import (
    PortfolioAnalysisResult,
    plot_weights,
    run_portfolio_analysis,
    summary_statistics,
    tangency_vs_comparison,
)
from .robustness import (
    VARIANTS,
    RobustnessResult,
    long_only_tangency,
    out_of_sample_sharpe,
    plot_robustness,
    run_robustness_analysis,
    shrink_covariance,
    shrink_means,
    tangency_variants,
    variant_stability,
    variant_weights,
    weight_diagnostics,
)

__all__ = [
    "capm",
    "custom_factors",
    "build_custom_factors",
    "factor_scorecard",
    "near_high_signal",
    "spanning_regression",
    "volatility_scale",
    "long_short_weights",
    "factor_construction",
    "factor_models",
    "portfolio",
    "portfolio_choice",
    "robustness",
    "out_of_sample",
    "OOS_BENCHMARKS",
    "OutOfSampleResult",
    "out_of_sample_summary",
    "run_out_of_sample_analysis",
    "walk_forward_returns",
    "FactorModelResult",
    "alpha_comparison",
    "build_self_factors",
    "factor_regression",
    "long_short_spread",
    "model_summary",
    "momentum_factor",
    "momentum_signal",
    "run_factor_analysis",
    "size_factor",
    "plot_sml",
    "plot_robustness",
    "VARIANTS",
    "RobustnessResult",
    "long_only_tangency",
    "out_of_sample_sharpe",
    "run_robustness_analysis",
    "shrink_covariance",
    "shrink_means",
    "tangency_variants",
    "variant_stability",
    "variant_weights",
    "weight_diagnostics",
    "CapmResult",
    "SMLFit",
    "empirical_sml",
    "excess_returns",
    "grs_test",
    "run_capm_analysis",
    "security_characteristic_line",
    "PortfolioAnalysisResult",
    "run_portfolio_analysis",
    "summary_statistics",
    "tangency_vs_comparison",
    "plot_weights",
    "equal_weights",
    "excess_means",
    "normalize_weights",
    "subsample_tangency",
    "tangency_weights",
    "weight_correlation",
]