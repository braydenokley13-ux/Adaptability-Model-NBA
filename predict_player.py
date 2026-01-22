#!/usr/bin/env python3
"""CLI tool for making adaptability predictions.

Usage:
    python predict_player.py "Player Name"          # Predict from training data
    python predict_player.py --list                 # List available players
    python predict_player.py --batch output.csv    # Predict all players

Examples:
    python predict_player.py "Tim Duncan"
    python predict_player.py "Steve Nash"
    python predict_player.py --list
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.prediction_engine import (
    AdaptabilityPredictor,
    format_prediction_report,
)
from src.utils import load_training_data


def list_available_players():
    """Print list of available players in training data."""
    df = load_training_data()
    players = sorted(df["Player"].unique())

    print("\nAvailable players in training data:")
    print("=" * 40)
    for player in players:
        tier = df[df["Player"] == player]["adaptability_tier"].iloc[0]
        tier_label = {0: "FADED", 1: "SURVIVED", 2: "THRIVED"}[tier]
        print(f"  {player:30s} (actual: {tier_label})")
    print(f"\nTotal: {len(players)} players")


def predict_single_player(player_name: str, predictor: AdaptabilityPredictor):
    """Make prediction for a single player."""
    df = load_training_data()

    if player_name not in df["Player"].values:
        # Try partial match
        matches = [p for p in df["Player"].values if player_name.lower() in p.lower()]
        if matches:
            print(f"Player '{player_name}' not found. Did you mean:")
            for m in matches[:5]:
                print(f"  - {m}")
            return
        else:
            print(f"Player '{player_name}' not found in training data.")
            print("Use --list to see available players.")
            return

    result = predictor.predict_from_dataframe(player_name, df)
    print(format_prediction_report(result))

    # Show actual outcome
    actual_tier = df[df["Player"] == player_name]["adaptability_tier"].iloc[0]
    actual_label = {0: "FADED", 1: "SURVIVED", 2: "THRIVED"}[actual_tier]
    print(f"\n📊 ACTUAL OUTCOME: {actual_label}")

    if actual_tier == result.predicted_tier:
        print("✅ Prediction correct!")
    else:
        print(f"❌ Misclassification (predicted {result.predicted_label}, was {actual_label})")


def batch_predict(output_path: str, predictor: AdaptabilityPredictor):
    """Run predictions on all players and save to CSV."""
    df = load_training_data()
    players = df["Player"].unique()

    results = []
    for player in players:
        try:
            result = predictor.predict_from_dataframe(player, df)
            actual_tier = df[df["Player"] == player]["adaptability_tier"].iloc[0]

            results.append({
                "Player": player,
                "Predicted_Tier": result.predicted_tier,
                "Predicted_Label": result.predicted_label,
                "Prob_FADED": result.probabilities["FADED (0)"],
                "Prob_SURVIVED": result.probabilities["SURVIVED (1)"],
                "Prob_THRIVED": result.probabilities["THRIVED (2)"],
                "Confidence": result.confidence,
                "Actual_Tier": actual_tier,
                "Actual_Label": {0: "FADED", 1: "SURVIVED", 2: "THRIVED"}[actual_tier],
                "Correct": result.predicted_tier == actual_tier,
            })
        except Exception as e:
            print(f"Error predicting {player}: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Prob_THRIVED", ascending=False)
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved predictions for {len(results)} players to {output_path}")

    # Summary stats
    correct = results_df["Correct"].sum()
    total = len(results_df)
    print(f"Accuracy: {correct}/{total} ({correct/total:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description="NBA Adaptability Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "player",
        nargs="?",
        help="Player name to predict (in quotes if contains spaces)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available players in training data",
    )
    parser.add_argument(
        "--batch",
        metavar="OUTPUT",
        help="Predict all players and save to CSV file",
    )

    args = parser.parse_args()

    if args.list:
        list_available_players()
        return

    if args.batch:
        print("Loading model...")
        predictor = AdaptabilityPredictor()
        print("Running batch predictions...")
        batch_predict(args.batch, predictor)
        return

    if args.player:
        print("Loading model...")
        predictor = AdaptabilityPredictor()
        predict_single_player(args.player, predictor)
        return

    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()
