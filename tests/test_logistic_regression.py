from pathlib import Path

import numpy as np
import pandas as pd

from domain.gradient_descent_strategy import GradientDescentStrategy
from src.domain.logistic_regression import LogisticRegressionModel


DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "dataset_train.csv"

# Numeric Hogwarts subject columns from the dataset schema.
FEATURE_COLUMNS = [
	# "Arithmancy",
	# "Astronomy",
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


def _load_dataframe() -> pd.DataFrame:
	return pd.read_csv(DATASET_PATH)


def _train_test_split_df(
	df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
	shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
	split_index = int(len(shuffled_df) * train_ratio)

	return shuffled_df.iloc[:split_index].copy(), shuffled_df.iloc[split_index:].copy()


def _prepare_ndarrays(
	train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	X_train_df = train_df[FEATURE_COLUMNS].copy()
	X_test_df = test_df[FEATURE_COLUMNS].copy()

	train_means = X_train_df.mean(numeric_only=True)
	X_train_df = X_train_df.fillna(train_means)
	X_test_df = X_test_df.fillna(train_means)

	X_train = X_train_df.to_numpy(dtype=float)
	X_test = X_test_df.to_numpy(dtype=float)
	y_train = train_df["Hogwarts House"].to_numpy(dtype=str)
	y_test = test_df["Hogwarts House"].to_numpy(dtype=str)

	return X_train, X_test, y_train, y_test


def test_batch_gd_logistic_regression_with_95_5_split() -> None:
	df = _load_dataframe()
	train_df, test_df = _train_test_split_df(df, train_ratio=0.95, seed=42)

	expected_train_size = int(len(df) * 0.95)
	expected_test_size = len(df) - expected_train_size
	assert len(train_df) == expected_train_size
	assert len(test_df) == expected_test_size

	X_train, X_test, y_train, y_test = _prepare_ndarrays(train_df, test_df)

	model = LogisticRegressionModel(learning_rate=0.05, num_iterations=1500)
	model.fit(X_train, y_train)

	predictions = model.predict(X_test)
	accuracy = LogisticRegressionModel.compare_predictions(predictions, y_test)

	assert predictions.shape == y_test.shape
	assert set(np.unique(predictions)).issubset(set(np.unique(y_train)))
	assert 0.0 <= accuracy <= 1.0
	assert accuracy >= 0.98

def test_minibatch_gd_logistic_regression_with_80_20_split() -> None:
	df = _load_dataframe()
	train_df, test_df = _train_test_split_df(df, train_ratio=0.8, seed=42)

	expected_train_size = int(len(df) * 0.8)
	expected_test_size = len(df) - expected_train_size
	assert len(train_df) == expected_train_size
	assert len(test_df) == expected_test_size

	X_train, X_test, y_train, y_test = _prepare_ndarrays(train_df, test_df)

	model = LogisticRegressionModel(
		learning_rate=0.05,
		num_iterations=500,
		gd_strategy=GradientDescentStrategy.MINI_BATCH(batch_size=16),
	)
	model.fit(X_train, y_train)

	predictions = model.predict(X_test)
	accuracy = LogisticRegressionModel.compare_predictions(predictions, y_test)

	assert predictions.shape == y_test.shape
	assert set(np.unique(predictions)).issubset(set(np.unique(y_train)))
	assert 0.00 <= accuracy <= 1.0
	assert accuracy >= 0.98

def test_stochastic_gd_logistic_regression_with_80_20_split() -> None:
	df = _load_dataframe()
	train_df, test_df = _train_test_split_df(df, train_ratio=0.8, seed=42)

	expected_train_size = int(len(df) * 0.8)
	expected_test_size = len(df) - expected_train_size
	assert len(train_df) == expected_train_size
	assert len(test_df) == expected_test_size

	X_train, X_test, y_train, y_test = _prepare_ndarrays(train_df, test_df)

	model = LogisticRegressionModel(
		learning_rate=0.05,
		num_iterations=100,
		gd_strategy=GradientDescentStrategy.STOCHASTIC(),
	)
	model.fit(X_train, y_train)

	predictions = model.predict(X_test)
	accuracy = LogisticRegressionModel.compare_predictions(predictions, y_test)

	assert predictions.shape == y_test.shape
	assert set(np.unique(predictions)).issubset(set(np.unique(y_train)))
	assert 0.00 <= accuracy <= 1.0
	assert accuracy >= 0.98