"""Predict Hogwarts houses using trained weights and output houses.csv."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
DEFAULT_OUTPUT = Path("houses.csv")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Hogwarts houses.")
    parser.add_argument("dataset", type=Path, help="Path to dataset_test.csv")
    parser.add_argument("weights", type=Path, help="Path to weights JSON file")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args()


def _prepare_array(df: pd.DataFrame, impute_means: np.ndarray) -> np.ndarray:
    """Extract features, imputing missing values with the model's training means."""
    X_df = df[FEATURE_COLUMNS].copy()
    X_df = X_df.fillna(dict(zip(FEATURE_COLUMNS, impute_means)))
    return X_df.to_numpy(dtype=float)


def main() -> None:
    parser = _build_parser()
    args = _parse_args(parser)

    model = LogisticRegressionModel.from_json(str(args.weights))

    df = pd.read_csv(args.dataset)
    # Reuse training means (stored in model) for imputation — keeps predict/train consistent
    X = _prepare_array(df, model.mean)

    predictions = model.predict(X)

    output_df = pd.DataFrame({"Index": range(len(predictions)), "Hogwarts House": predictions})
    output_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print("\nAvailable options:")
    print(parser.format_help())


if __name__ == "__main__":
    main()
