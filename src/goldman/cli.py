"""Command-line entry point.

Examples
--------
Run the mean–variance portfolio analysis on the real US universe, using the
FF3 risk-free rate::

    goldman portfolios --universe us --use-ff3-rf

Run the same analysis but force the 1/N comparison instead of value weights::

    goldman portfolios --universe us --equal-weights

Run every implemented analysis on the synthetic sample::

    goldman all --universe synthetic --output-dir outputs/smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config
from .analysis import portfolio_choice
from .data import load_ff3, load_returns, load_value_weights, monthly_rf

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goldman", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--universe",
            choices=sorted(config.UNIVERSES),
            default="synthetic",
            help="Named price universe (default: synthetic).",
        )
        p.add_argument("--prices", type=Path, default=None, help="Explicit price CSV path.")
        p.add_argument(
            "--min-obs",
            type=int,
            default=config.MIN_OBS,
            help=f"Minimum clean observations per asset (default: {config.MIN_OBS}).",
        )
        p.add_argument(
            "--rf",
            type=float,
            default=config.RF_MONTHLY,
            help=f"Fallback monthly risk-free rate (default: {config.RF_MONTHLY}).",
        )
        p.add_argument(
            "--use-ff3-rf",
            action="store_true",
            help="Use the monthly RF column from ff3_monthly.csv when available.",
        )
        p.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Where exhibits are written (default: <repo>/outputs).",
        )
        p.add_argument(
            "--value-weights",
            type=Path,
            default=None,
            help=(
                "CSV with an 'asset' column and 'weight' or 'market_cap' "
                "(default: the registered value-weight file for --universe)."
            ),
        )
        p.add_argument(
            "--equal-weights",
            action="store_true",
            help="Force the 1/N portfolio as the comparison instead of value weights.",
        )

    portfolios = sub.add_parser(
        "portfolios",
        help="Mean–variance portfolio choice and estimation error.",
    )
    add_common(portfolios)

    run_all = sub.add_parser("all", help="Run every implemented analysis end to end.")
    add_common(run_all)
    return parser


def _resolve_returns(args: argparse.Namespace) -> pd.DataFrame:
    if args.prices is not None:
        return load_returns(path=args.prices, min_obs=args.min_obs)
    return load_returns(universe=args.universe, min_obs=args.min_obs)


def _resolve_rf(args: argparse.Namespace, returns: pd.DataFrame) -> float | pd.Series:
    if args.use_ff3_rf:
        ff3 = load_ff3()
        return monthly_rf(returns.index, ff3=ff3, fallback=args.rf)
    return args.rf


def _resolve_comparison(
    args: argparse.Namespace, returns: pd.DataFrame
) -> tuple[pd.Series | pd.DataFrame | None, str]:
    """Pick the comparison weights and a label for the chart.

    Priority: explicit ``--value-weights`` file, then the registered
    value-weight file for the chosen universe, else the 1/N portfolio.  For a
    date-by-asset value-weight panel the chart uses the sample-average weights
    (the full monthly panel is written to the output directory).
    """
    dates = returns.index
    if args.value_weights is not None:
        return (
            load_value_weights(path=args.value_weights, assets=returns.columns, dates=dates),
            "value weights",
        )
    if not args.equal_weights:
        registered = config.value_weights_path(args.universe)
        if registered is not None and registered.exists():
            weights = load_value_weights(
                path=registered, assets=returns.columns, dates=dates
            )
            return weights, "value weights (sample average)"
    return None, "1/N"


def _run_portfolio_analysis(args: argparse.Namespace) -> int:
    returns = _resolve_returns(args)
    rf = _resolve_rf(args, returns)
    comparison, comparison_label = _resolve_comparison(args, returns)

    print(
        f"Universe: {returns.shape[1]} assets, {returns.shape[0]} months "
        f"({returns.index[0]:%Y-%m} .. {returns.index[-1]:%Y-%m})"
    )
    print(f"Comparison portfolio: {comparison_label}")

    result = portfolio_choice.run_portfolio_analysis(
        returns,
        rf=rf,
        comparison=comparison,
        comparison_label=comparison_label,
        output_dir=args.output_dir,
    )

    print("Summary statistics written:", result.output_paths["summary"])
    print(
        "Weight chart written:",
        result.output_paths["weights"],
        f"(most extreme tangency position: {result.most_extreme_weight:+.0%})",
    )
    if "value_weights" in result.output_paths:
        print("Point-in-time value weights written:", result.output_paths["value_weights"])
    print(
        "Subsample weights written:",
        result.output_paths["subsample"],
        f"(corr between half-sample weight vectors: {result.weight_correlation:.2f})",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in {"portfolios", "all"}:
        return _run_portfolio_analysis(args)

    parser.error(f"Unknown command: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())