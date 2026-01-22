"""Data pipeline for NBA Adaptability Model.

This module handles loading, cleaning, and preprocessing the raw NBA season stats
data for use in the adaptability prediction model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# Default path to raw data
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "Seasons_Stats.csv"

# Key statistical columns used for features
CORE_STATS = ["USG%", "AST%", "3PAr", "TS%", "BPM", "WS/48", "MP", "PER"]

# Additional context columns
CONTEXT_COLS = ["Year", "Player", "Pos", "Age", "Tm", "G", "GS"]

# Minimum minutes threshold for a qualifying season
MIN_MINUTES = 500

# Year range for modern era (when advanced stats are reliable)
MIN_YEAR = 2005
MAX_YEAR = 2017  # Limited by current dataset


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load raw season stats CSV data.

    Args:
        filepath: Path to CSV file. Uses default path if None.

    Returns:
        Raw DataFrame with all seasons.
    """
    if filepath is None:
        filepath = DEFAULT_DATA_PATH

    df = pd.read_csv(filepath)

    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def _deduplicate_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Handle players traded mid-season by preferring TOT (total) rows.

    When a player plays for multiple teams in a season, Basketball Reference
    creates separate rows plus a 'TOT' row for combined stats. We prefer TOT.
    """
    if "Tm" not in df.columns:
        return df

    df = df.copy()

    # Create a priority column: TOT gets priority 0, others get priority 1
    df["_priority"] = df["Tm"].apply(lambda x: 0 if x == "TOT" else 1)

    # Sort by player, year, priority (TOT first), then minutes (higher first)
    df = df.sort_values(
        ["Player", "Year", "_priority", "MP"],
        ascending=[True, True, True, False]
    )

    # Keep first row for each player-year combination (which will be TOT if exists, else highest MP)
    df = df.drop_duplicates(subset=["Player", "Year"], keep="first")

    # Remove the helper column
    df = df.drop(columns=["_priority"])

    return df.reset_index(drop=True)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert key columns to numeric, handling any string values."""
    numeric_cols = ["Age", "Year", "MP", "G", "GS"] + CORE_STATS

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _clean_player_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize player names by removing special characters and suffixes."""
    # Remove asterisks (Hall of Fame markers)
    df["Player"] = df["Player"].str.replace("*", "", regex=False)
    # Strip whitespace
    df["Player"] = df["Player"].str.strip()
    return df


def clean_data(df: pd.DataFrame, min_year: int = MIN_YEAR) -> pd.DataFrame:
    """Apply all cleaning operations to raw data.

    Args:
        df: Raw DataFrame.
        min_year: Minimum year to include (default 2005 for reliable advanced stats).

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()

    # Apply cleaning steps
    df = _clean_player_names(df)
    df = _coerce_numeric_columns(df)
    df = _deduplicate_team_rows(df)

    # Filter to modern era with reliable stats
    df = df[df["Year"] >= min_year]
    df = df[df["Year"] <= MAX_YEAR]

    # Filter out seasons with too few minutes
    df = df[df["MP"] >= MIN_MINUTES]

    # Drop rows with missing key stats
    required_cols = ["Player", "Year", "Age", "MP"]
    df = df.dropna(subset=required_cols)

    # Sort by player and year
    df = df.sort_values(["Player", "Year"]).reset_index(drop=True)

    return df


def identify_eligible_players(
    df: pd.DataFrame,
    age_30_year_min: int = 2005,
    age_30_year_max: int = 2012
) -> pd.DataFrame:
    """Identify players eligible for training based on age-30 season timing.

    Players must have turned 30 within the specified year range to allow
    observation of their 35+ performance within the dataset.

    Args:
        df: Cleaned DataFrame.
        age_30_year_min: Earliest year a player can have their age-30 season.
        age_30_year_max: Latest year a player can have their age-30 season.

    Returns:
        DataFrame with eligible players' age-30 season info.
    """
    # Find age-30 seasons with sufficient minutes
    age_30_seasons = df[(df["Age"] == 30) & (df["MP"] >= MIN_MINUTES)].copy()

    # Filter to years where we can observe 35+ outcomes
    eligible = age_30_seasons[
        (age_30_seasons["Year"] >= age_30_year_min) &
        (age_30_seasons["Year"] <= age_30_year_max)
    ]

    return eligible[["Player", "Year"]].drop_duplicates()


def get_player_seasons(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Get all seasons for a specific player.

    Args:
        df: Full dataset.
        player_name: Player's name to filter.

    Returns:
        DataFrame with only that player's seasons, sorted by year.
    """
    player_df = df[df["Player"] == player_name].copy()
    return player_df.sort_values("Year").reset_index(drop=True)


def load_and_clean_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to load and clean data in one step.

    Args:
        filepath: Path to CSV file. Uses default if None.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    raw_df = load_raw_data(filepath)
    clean_df = clean_data(raw_df)
    return clean_df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics about the cleaned dataset.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Dictionary with summary statistics.
    """
    return {
        "total_records": len(df),
        "unique_players": df["Player"].nunique(),
        "year_range": (int(df["Year"].min()), int(df["Year"].max())),
        "age_range": (int(df["Age"].min()), int(df["Age"].max())),
        "avg_minutes": df["MP"].mean(),
        "stats_availability": {
            stat: df[stat].notna().sum() / len(df) * 100
            for stat in CORE_STATS if stat in df.columns
        }
    }


if __name__ == "__main__":
    # Test the pipeline
    print("Loading and cleaning data...")
    df = load_and_clean_data()

    print("\nData Summary:")
    summary = get_data_summary(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nIdentifying eligible players...")
    eligible = identify_eligible_players(df)
    print(f"  Found {len(eligible)} eligible players")
    print(f"  Sample: {eligible['Player'].head(10).tolist()}")
