
"""
AgriSmart - Dataset Preparation Module

Loads the raw crop recommendation dataset,
validates it, and creates training/testing datasets.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Define project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 2. Define dataset configuration
TARGET_COLUMN = "label"

FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset():
    """Load and validate the original dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if not data[required_columns].select_dtypes(
        include="number"
    ).columns.tolist() == FEATURE_COLUMNS:
        raise ValueError("One or more input features are not numeric.")

    if not pd.api.types.is_numeric_dtype(data[TARGET_COLUMN]):
        pass  # Crop labels are expected to be text.

    return data


def prepare_data():
    """Separate features and target, then split the dataset."""

    data = load_dataset()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("\nAgriSmart Dataset Preparation Completed")
    print("-" * 45)
    print(f"Original dataset shape: {data.shape}")
    print(f"Training features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")
    print(f"Training records: {len(y_train)}")
    print(f"Testing records: {len(y_test)}")
    print(f"Number of crop classes: {y.nunique()}")
    print("\nFiles saved successfully in data/processed/")


if __name__ == "__main__":
    prepare_data()