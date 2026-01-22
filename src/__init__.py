"""NBA Adaptability Model - ML pipeline for predicting player longevity."""

__version__ = "1.0.0"

from .data_pipeline import load_and_clean_data, identify_eligible_players
from .feature_engineering import (
    build_training_dataset,
    extract_all_features,
    label_player,
    TIER_LABELS,
)
from .utils import (
    load_training_data,
    load_train_test_split,
    get_feature_columns,
)

__all__ = [
    "load_and_clean_data",
    "identify_eligible_players",
    "build_training_dataset",
    "extract_all_features",
    "label_player",
    "TIER_LABELS",
    "load_training_data",
    "load_train_test_split",
    "get_feature_columns",
]
