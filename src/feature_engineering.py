"""Feature engineering for NBA Adaptability Model.

This module extracts features from player seasons (ages 30-33) and labels
players based on their performance at age 35+.

Target Variable (Adaptability Tier):
    2 = THRIVED: High-impact player into late 30s
    1 = SURVIVED: Valuable rotation player, adapted to reduced role
    0 = FADED: Could not adapt, out of league or minimal impact
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_pipeline import CORE_STATS, load_and_clean_data, identify_eligible_players


# Feature extraction ages
BASELINE_AGE = 30
DELTA_AGES = [31, 32, 33]
OUTCOME_AGE = 35

# Tier labels
TIER_LABELS = {0: "FADED", 1: "SURVIVED", 2: "THRIVED"}


def _get_season_at_age(player_df: pd.DataFrame, age: int) -> Optional[pd.Series]:
    """Get a player's season stats at a specific age.

    Args:
        player_df: DataFrame with one player's seasons.
        age: Target age.

    Returns:
        Series with season stats, or None if no qualifying season.
    """
    season = player_df[player_df["Age"] == age]
    if season.empty:
        return None
    # Take most recent if multiple (shouldn't happen after dedup)
    return season.sort_values("Year").iloc[-1]


def extract_baseline_features(player_df: pd.DataFrame) -> Dict[str, float]:
    """Extract baseline statistics at age 30.

    Args:
        player_df: DataFrame with one player's seasons.

    Returns:
        Dictionary of baseline features.
    """
    features = {}
    age_30 = _get_season_at_age(player_df, BASELINE_AGE)

    if age_30 is None:
        # Return NaN for all baseline features
        for stat in CORE_STATS:
            features[f"age30_{stat.replace('%', 'pct').replace('/', '_')}"] = np.nan
        return features

    for stat in CORE_STATS:
        # Clean up stat name for column naming
        clean_name = stat.replace("%", "pct").replace("/", "_")
        features[f"age30_{clean_name}"] = age_30.get(stat, np.nan)

    return features


def extract_delta_features(player_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate change metrics from age 30 to ages 31, 32, 33.

    Args:
        player_df: DataFrame with one player's seasons.

    Returns:
        Dictionary of delta features.
    """
    features = {}
    age_30 = _get_season_at_age(player_df, BASELINE_AGE)

    for target_age in DELTA_AGES:
        age_row = _get_season_at_age(player_df, target_age)

        for stat in CORE_STATS:
            clean_name = stat.replace("%", "pct").replace("/", "_")
            feature_name = f"delta_{clean_name}_{target_age}"

            if age_30 is None or age_row is None:
                features[feature_name] = np.nan
            else:
                baseline_val = age_30.get(stat, np.nan)
                target_val = age_row.get(stat, np.nan)

                if pd.notna(baseline_val) and pd.notna(target_val):
                    features[feature_name] = target_val - baseline_val
                else:
                    features[feature_name] = np.nan

    return features


def extract_career_context_features(player_df: pd.DataFrame) -> Dict[str, float]:
    """Extract career context features.

    Args:
        player_df: DataFrame with one player's seasons.

    Returns:
        Dictionary of career context features.
    """
    features = {}

    # Count seasons by age 30
    seasons_before_30 = player_df[player_df["Age"] <= 30]
    features["seasons_at_30"] = len(seasons_before_30)

    # Career high usage rate (before age 30)
    if "USG%" in player_df.columns:
        early_career = player_df[player_df["Age"] < 30]
        if not early_career.empty and early_career["USG%"].notna().any():
            features["career_high_USG"] = early_career["USG%"].max()
        else:
            features["career_high_USG"] = np.nan
    else:
        features["career_high_USG"] = np.nan

    # Career average BPM (before 30)
    if "BPM" in player_df.columns:
        early_career = player_df[player_df["Age"] < 30]
        if not early_career.empty and early_career["BPM"].notna().any():
            features["career_avg_BPM"] = early_career["BPM"].mean()
        else:
            features["career_avg_BPM"] = np.nan
    else:
        features["career_avg_BPM"] = np.nan

    # Position archetype (simplified)
    age_30 = _get_season_at_age(player_df, BASELINE_AGE)
    if age_30 is not None:
        pos = age_30.get("Pos", "")
        # Map positions to numeric codes
        if pd.notna(pos):
            pos = str(pos)
            if "C" in pos:
                features["position_code"] = 5  # Center
            elif "PF" in pos:
                features["position_code"] = 4  # Power Forward
            elif "SF" in pos:
                features["position_code"] = 3  # Small Forward
            elif "SG" in pos:
                features["position_code"] = 2  # Shooting Guard
            elif "PG" in pos:
                features["position_code"] = 1  # Point Guard
            else:
                features["position_code"] = 3  # Default to SF
        else:
            features["position_code"] = np.nan
    else:
        features["position_code"] = np.nan

    return features


def extract_cumulative_trend_features(player_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate cumulative trend features from age 30 to 33.

    Args:
        player_df: DataFrame with one player's seasons.

    Returns:
        Dictionary of cumulative trend features.
    """
    features = {}
    age_30 = _get_season_at_age(player_df, BASELINE_AGE)
    age_33 = _get_season_at_age(player_df, 33)

    # Total changes 30 -> 33
    for stat in ["3PAr", "USG%", "TS%", "AST%"]:
        clean_name = stat.replace("%", "pct").replace("/", "_")
        feature_name = f"total_{clean_name}_change"

        if age_30 is None or age_33 is None:
            features[feature_name] = np.nan
        else:
            val_30 = age_30.get(stat, np.nan)
            val_33 = age_33.get(stat, np.nan)

            if pd.notna(val_30) and pd.notna(val_33):
                features[feature_name] = val_33 - val_30
            else:
                features[feature_name] = np.nan

    # Adaptation velocity: average rate of change per year
    if pd.notna(features.get("total_USGpct_change")):
        features["adaptation_velocity_USG"] = features["total_USGpct_change"] / 3
    else:
        features["adaptation_velocity_USG"] = np.nan

    if pd.notna(features.get("total_3PAr_change")):
        features["adaptation_velocity_3PAr"] = features["total_3PAr_change"] / 3
    else:
        features["adaptation_velocity_3PAr"] = np.nan

    return features


def label_player(player_df: pd.DataFrame) -> int:
    """Determine adaptability tier based on age 35+ performance.

    Tier Definitions:
        2 (THRIVED): 3+ seasons at 35+ AND (avg WS/48 >= 0.100 OR avg BPM >= 1.5) AND avg MP >= 1200
        1 (SURVIVED): 2+ seasons at 35+ AND (avg WS/48 >= 0.050 OR avg BPM >= 0.0) AND avg MP >= 600
        0 (FADED): Does not meet SURVIVED criteria

    Args:
        player_df: DataFrame with one player's seasons.

    Returns:
        Tier label (0, 1, or 2).
    """
    # Get all seasons at age 35+
    age_35_plus = player_df[player_df["Age"] >= OUTCOME_AGE]

    # Count qualifying seasons (at least some minutes played)
    num_seasons = len(age_35_plus)

    if num_seasons < 2:
        return 0  # FADED - didn't make it or had only 1 season

    # Calculate averages
    avg_ws48 = age_35_plus["WS/48"].mean() if "WS/48" in age_35_plus.columns else np.nan
    avg_bpm = age_35_plus["BPM"].mean() if "BPM" in age_35_plus.columns else np.nan
    avg_mp = age_35_plus["MP"].mean() if "MP" in age_35_plus.columns else np.nan

    # Handle NaN values
    if pd.isna(avg_ws48):
        avg_ws48 = -999
    if pd.isna(avg_bpm):
        avg_bpm = -999
    if pd.isna(avg_mp):
        avg_mp = 0

    # Check THRIVED criteria
    if num_seasons >= 3 and (avg_ws48 >= 0.100 or avg_bpm >= 1.5) and avg_mp >= 1200:
        return 2  # THRIVED

    # Check SURVIVED criteria
    if num_seasons >= 2 and (avg_ws48 >= 0.050 or avg_bpm >= 0.0) and avg_mp >= 600:
        return 1  # SURVIVED

    return 0  # FADED


def extract_all_features(player_df: pd.DataFrame) -> Dict[str, float]:
    """Extract all features for a single player.

    Args:
        player_df: DataFrame with one player's seasons.

    Returns:
        Dictionary with all features combined.
    """
    features = {}
    features.update(extract_baseline_features(player_df))
    features.update(extract_delta_features(player_df))
    features.update(extract_career_context_features(player_df))
    features.update(extract_cumulative_trend_features(player_df))
    return features


def build_training_dataset(
    df: pd.DataFrame,
    eligible_players: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build complete training dataset with features and labels.

    Args:
        df: Full cleaned dataset.
        eligible_players: DataFrame with Player and Year columns for eligible players.

    Returns:
        Tuple of (features_df, labels_series, player_names_series).
    """
    feature_rows = []
    labels = []
    players = []

    for _, row in eligible_players.iterrows():
        player_name = row["Player"]
        player_df = df[df["Player"] == player_name].copy()

        if player_df.empty:
            continue

        # Extract features
        features = extract_all_features(player_df)

        # Get label
        label = label_player(player_df)

        feature_rows.append(features)
        labels.append(label)
        players.append(player_name)

    features_df = pd.DataFrame(feature_rows)
    labels_series = pd.Series(labels, name="adaptability_tier")
    players_series = pd.Series(players, name="Player")

    return features_df, labels_series, players_series


def get_feature_names() -> List[str]:
    """Get list of all feature names in order."""
    # Build a sample feature dict to get column names
    sample_features = {}

    # Baseline features
    for stat in CORE_STATS:
        clean_name = stat.replace("%", "pct").replace("/", "_")
        sample_features[f"age30_{clean_name}"] = 0

    # Delta features
    for age in DELTA_AGES:
        for stat in CORE_STATS:
            clean_name = stat.replace("%", "pct").replace("/", "_")
            sample_features[f"delta_{clean_name}_{age}"] = 0

    # Career context
    sample_features["seasons_at_30"] = 0
    sample_features["career_high_USG"] = 0
    sample_features["career_avg_BPM"] = 0
    sample_features["position_code"] = 0

    # Cumulative trends
    for stat in ["3PAr", "USG%", "TS%", "AST%"]:
        clean_name = stat.replace("%", "pct").replace("/", "_")
        sample_features[f"total_{clean_name}_change"] = 0

    sample_features["adaptation_velocity_USG"] = 0
    sample_features["adaptation_velocity_3PAr"] = 0

    return list(sample_features.keys())


def describe_tier(tier: int) -> str:
    """Get human-readable description for a tier."""
    descriptions = {
        0: "FADED - Could not adapt to aging, out of league or minimal impact after 35",
        1: "SURVIVED - Adapted to reduced role, remained valuable rotation player after 35",
        2: "THRIVED - Remained high-impact player, starter-quality into late 30s"
    }
    return descriptions.get(tier, "Unknown")


if __name__ == "__main__":
    import os

    print("Building training dataset...")

    # Load and clean data
    df = load_and_clean_data()
    eligible = identify_eligible_players(df)

    print(f"Processing {len(eligible)} eligible players...")

    # Build dataset
    features_df, labels, players = build_training_dataset(df, eligible)

    print(f"\nDataset built:")
    print(f"  Features shape: {features_df.shape}")
    print(f"  Feature columns: {features_df.columns.tolist()}")

    print(f"\nLabel distribution:")
    for tier in [0, 1, 2]:
        count = (labels == tier).sum()
        pct = count / len(labels) * 100
        print(f"  {TIER_LABELS[tier]} ({tier}): {count} ({pct:.1f}%)")

    # Show some example players per tier
    print("\nExample players by tier:")
    for tier in [2, 1, 0]:
        tier_players = players[labels == tier].tolist()[:5]
        print(f"  {TIER_LABELS[tier]}: {tier_players}")

    # Save dataset
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Combine into single dataframe
    full_df = features_df.copy()
    full_df["Player"] = players.values
    full_df["adaptability_tier"] = labels.values

    output_path = os.path.join(output_dir, "training_data.csv")
    full_df.to_csv(output_path, index=False)
    print(f"\nSaved training data to: {output_path}")
