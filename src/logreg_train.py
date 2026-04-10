"""Train a logistic regression model on the Hogwarts dataset and save weights to JSON."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from domain.gradient_descent_strategy import GradientDescentStrategy
from domain.logistic_regression import LogisticRegressionModel

FEATURE_COLUMNS = [
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
]
TARGET_COLUMN = "Hogwarts House"
DEFAULT_OUTPUT = Path("weights.json")


def _normalize_strategy_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "batch": "batch",
        "mini-batch": "mini-batch",
        "minibatch": "mini-batch",
        "stochastic": "stochastic",
    }

    if normalized not in aliases:
        allowed = "batch, mini-batch (or minibatch), stochastic"
        raise argparse.ArgumentTypeError(
            f"Invalid gradient descent strategy '{value}'. Choose one of: {allowed}."
        )

    return aliases[normalized]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train logistic regression on Hogwarts dataset.")
    parser.add_argument("dataset", type=Path, help="Path to dataset_train.csv")
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output weights file"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=1500, help="Number of iterations")
    parser.add_argument(
        "--gd-strategy",
        type=_normalize_strategy_name,
        default="batch",
        help="Gradient descent strategy: batch, mini-batch, or stochastic",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used only with mini-batch strategy",
    )

    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:

    args = parser.parse_args()

    if args.gd_strategy == "mini-batch" and args.batch_size < 1:
        parser.error("--batch-size must be >= 1 when --gd-strategy mini-batch is used")

    return args


def _resolve_gd_strategy(strategy_name: str, batch_size: int) -> GradientDescentStrategy:
    if strategy_name == "batch":
        return GradientDescentStrategy.BATCH()
    if strategy_name == "stochastic":
        return GradientDescentStrategy.STOCHASTIC()

    return GradientDescentStrategy.MINI_BATCH(batch_size=batch_size)


def _prepare_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and target, imputing missing values with column means."""
    X_df = df[FEATURE_COLUMNS].copy()
    col_means = X_df.mean(numeric_only=True)
    X_df = X_df.fillna(col_means)

    X = X_df.to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=str)
    return X, y, col_means.to_numpy()


def main() -> None:
    parser = _build_parser()
    args = _parse_args(parser)

    df = pd.read_csv(args.dataset)
    X, y, _ = _prepare_arrays(df)
    gd_strategy = _resolve_gd_strategy(args.gd_strategy, args.batch_size)

    model = LogisticRegressionModel(
        learning_rate=args.lr,
        num_iterations=args.iterations,
        gd_strategy=gd_strategy,
    )
    model.fit(X, y)
    model.save_json(str(args.output))
    print(f"Model saved to {args.output}")
    print("\nAvailable options:")
    print(parser.format_help())


if __name__ == "__main__":
    main()
