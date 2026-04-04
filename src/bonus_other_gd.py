"""Compare Batch, Mini-Batch, and Stochastic gradient descent strategies."""

import time
from pathlib import Path

import pandas as pd

from src.domain.gradient_descent_strategy import GradientDescentStrategy
from src.domain.logistic_regression import LogisticRegressionModel

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "dataset_train.csv"
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

STRATEGIES = [
    ("Batch GD", GradientDescentStrategy.BATCH(), 0.05, 1500),
    ("Mini-Batch GD", GradientDescentStrategy.MINI_BATCH(batch_size=16), 0.05, 500),
    ("Stochastic GD", GradientDescentStrategy.STOCHASTIC(), 0.05, 100),
]


def _load_and_split(path: Path, train_ratio: float = 0.8, seed: int = 42):
    df = pd.read_csv(path).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split = int(len(df) * train_ratio)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    X_train_df = train_df[FEATURE_COLUMNS].copy()
    col_means = X_train_df.mean(numeric_only=True)
    X_train_df = X_train_df.fillna(col_means)
    X_test_df = test_df[FEATURE_COLUMNS].fillna(col_means)

    return (
        X_train_df.to_numpy(dtype=float),
        X_test_df.to_numpy(dtype=float),
        train_df[TARGET_COLUMN].to_numpy(dtype=str),
        test_df[TARGET_COLUMN].to_numpy(dtype=str),
    )


def main() -> None:
    X_train, X_test, y_train, y_test = _load_and_split(DATASET_PATH)

    print(f"\n{'Strategy':<18} {'Accuracy':>10} {'Time (s)':>10}")
    print("-" * 42)

    for name, strategy, lr, iterations in STRATEGIES:
        model = LogisticRegressionModel(
            learning_rate=lr,
            num_iterations=iterations,
            gd_strategy=strategy,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        accuracy = LogisticRegressionModel.compare_predictions(model.predict(X_test), y_test)
        status = "✓" if accuracy >= 0.98 else "✗"
        print(f"{name:<18} {accuracy:>9.2%} {elapsed:>9.2f}s  {status}")

    print()


if __name__ == "__main__":
    main()
