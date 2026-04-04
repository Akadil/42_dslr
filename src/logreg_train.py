"""Train a logistic regression model on the Hogwarts dataset and save weights to JSON."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain.gradient_descent_strategy import GradientDescentStrategy
from src.domain.logistic_regression import LogisticRegressionModel

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic regression on Hogwarts dataset.")
    parser.add_argument("dataset", type=Path, help="Path to dataset_train.csv")
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output weights file"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=1500, help="Number of iterations")
    return parser.parse_args()


def _prepare_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and target, imputing missing values with column means."""
    X_df = df[FEATURE_COLUMNS].copy()
    col_means = X_df.mean(numeric_only=True)
    X_df = X_df.fillna(col_means)

    X = X_df.to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=str)
    return X, y, col_means.to_numpy()


def main() -> None:
    args = _parse_args()

    df = pd.read_csv(args.dataset)
    X, y, _ = _prepare_arrays(df)

    model = LogisticRegressionModel(
        learning_rate=args.lr,
        num_iterations=args.iterations,
        gd_strategy=GradientDescentStrategy.BATCH(),
    )
    model.fit(X, y)
    model.save_json(str(args.output))
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
