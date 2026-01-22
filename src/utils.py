"""Utility functions for NBA Adaptability Model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIZ_DIR = PROJECT_ROOT / "visualizations"


def ensure_directories():
    """Create project directories if they don't exist."""
    for dir_path in [PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, VIZ_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def load_training_data() -> pd.DataFrame:
    """Load the full training dataset.

    Returns:
        DataFrame with features, Player, and adaptability_tier columns.
    """
    path = PROCESSED_DIR / "training_data.csv"
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns (excludes Player and target)."""
    exclude = ["Player", "adaptability_tier"]
    return [col for col in df.columns if col not in exclude]


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets with stratification.

    Args:
        df: Full training DataFrame.
        test_size: Proportion for test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    feature_cols = get_feature_columns(df)

    # Stratify by label to maintain class distribution
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["adaptability_tier"]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_train_test_split(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save train and test sets to CSV files."""
    ensure_directories()

    train_path = PROCESSED_DIR / "train_set.csv"
    test_path = PROCESSED_DIR / "test_set.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train set ({len(train_df)} rows) to: {train_path}")
    print(f"Saved test set ({len(test_df)} rows) to: {test_path}")


def load_train_test_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously saved train and test sets.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_path = PROCESSED_DIR / "train_set.csv"
    test_path = PROCESSED_DIR / "test_set.csv"

    return pd.read_csv(train_path), pd.read_csv(test_path)


if __name__ == "__main__":
    print("Creating train/test split...")

    # Load training data
    df = load_training_data()
    print(f"Loaded {len(df)} total samples")

    # Create split
    train_df, test_df = create_train_test_split(df)

    print(f"\nTrain set: {len(train_df)} samples")
    print(f"  Class distribution:")
    for tier in [0, 1, 2]:
        count = (train_df["adaptability_tier"] == tier).sum()
        print(f"    Tier {tier}: {count}")

    print(f"\nTest set: {len(test_df)} samples")
    print(f"  Class distribution:")
    for tier in [0, 1, 2]:
        count = (test_df["adaptability_tier"] == tier).sum()
        print(f"    Tier {tier}: {count}")

    # Save
    save_train_test_split(train_df, test_df)
