"""Command-line entry point.

Examples
--------
Run the mean–variance portfolio analysis on the real US universe, using the
FF3 risk-free rate::

    goldman portfolios --universe us --use-ff3-rf

Run every implemented analysis on the synthetic sample::

    goldman all --universe synthetic --output-dir outputs/smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config
from .data import load_ff3, load_returns, monthly_rf
from .exhibits import portfolio_choice

__all__ = ["main"]


def _load_value_weights(path: Path, assets: pd.Index) -> pd.Series:
    """Load a two-column ``asset,weight`` CSV and align it to ``assets``."""
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"{path} must have at least two columns (asset, weight).")
    frame = frame.iloc[:, :2]
    frame.columns = ["asset", "weight"]
    series = frame.set_index("asset")["weight"].astype(float)
    series = series.reindex(assets)
    if series.isna().any():
        missing = ", ".join(series[series.isna()].index)
        raise ValueError(f"Value weights missing for: {missing}")
    return series / series.sum()


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
            help="Two-column CSV (asset,weight) to use as the comparison weights.",
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


def _run_portfolio_analysis(args: argparse.Namespace) -> int:
    returns = _resolve_returns(args)
    rf = _resolve_rf(args, returns)

    comparison = None
    if args.value_weights is not None:
        comparison = _load_value_weights(args.value_weights, returns.columns)

    print(
        f"Universe: {returns.shape[1]} assets, {returns.shape[0]} months "
        f"({returns.index[0]:%Y-%m} .. {returns.index[-1]:%Y-%m})"
    )

    result = portfolio_choice.run_portfolio_analysis(
        returns, rf=rf, comparison=comparison, output_dir=args.output_dir
    )

    print("Summary statistics written:", result.output_paths["summary"])
    print(
        "Weight chart written:",
        result.output_paths["weights"],
        f"(most extreme tangency position: {result.most_extreme_weight:+.0%})",
    )
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