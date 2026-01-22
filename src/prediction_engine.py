"""Prediction engine for NBA Adaptability Model.

This module provides the main prediction interface with explanations,
comparable players, and natural language interpretations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .model_training import load_model, TIER_LABELS
from .utils import get_feature_columns, load_training_data


@dataclass
class PredictionResult:
    """Container for prediction results with explanations."""

    player_name: str
    predicted_tier: int
    predicted_label: str
    probabilities: Dict[str, float]
    confidence: str
    top_positive_factors: List[Tuple[str, float]]
    top_negative_factors: List[Tuple[str, float]]
    comparable_players: List[Dict]
    interpretation: str


class AdaptabilityPredictor:
    """Main prediction class with explanations and comparisons."""

    def __init__(self):
        """Initialize predictor by loading model and training data."""
        self.pipeline, self.feature_columns, self.metadata = load_model()
        self.training_data = load_training_data()
        self._setup_comparable_players()

    def _setup_comparable_players(self):
        """Prepare k-NN model for finding comparable players."""
        # Get feature matrix from training data
        X_train = self.training_data[self.feature_columns].copy()

        # Impute missing values (same as training pipeline)
        imputer = self.pipeline.named_steps["imputer"]
        X_imputed = imputer.transform(X_train)

        # Scale features
        scaler = self.pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X_imputed)

        # Fit k-NN
        self.knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        self.knn.fit(X_scaled)

        # Store scaled features for lookup
        self.X_scaled_train = X_scaled

    def _get_confidence_level(self, probabilities: np.ndarray) -> str:
        """Determine confidence level from probability distribution."""
        max_prob = probabilities.max()
        if max_prob >= 0.6:
            return "HIGH"
        elif max_prob >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_feature_contributions(
        self,
        features: pd.Series,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Extract top positive and negative contributing features.

        Uses feature importances combined with feature values relative to
        training data means.
        """
        rf = self.pipeline.named_steps["classifier"]
        importances = dict(zip(self.feature_columns, rf.feature_importances_))

        # Get training data statistics
        train_means = self.training_data[self.feature_columns].mean()
        train_stds = self.training_data[self.feature_columns].std()

        contributions = []
        for col in self.feature_columns:
            if pd.isna(features.get(col)):
                continue

            value = features[col]
            mean = train_means[col]
            std = train_stds[col] if train_stds[col] > 0 else 1

            # Z-score relative to training data
            z_score = (value - mean) / std

            # Contribution = importance * direction
            importance = importances.get(col, 0)
            contribution = importance * z_score * 100  # Scale for readability

            # Create human-readable description
            description = self._describe_feature(col, value, mean, z_score)
            contributions.append((description, contribution, col))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Split into positive and negative
        positive = [(desc, val) for desc, val, _ in contributions if val > 0][:5]
        negative = [(desc, val) for desc, val, _ in contributions if val < 0][:5]

        return positive, negative

    def _describe_feature(
        self,
        feature_name: str,
        value: float,
        mean: float,
        z_score: float
    ) -> str:
        """Create human-readable description of a feature value."""
        direction = "above" if z_score > 0 else "below"
        magnitude = "significantly" if abs(z_score) > 1.5 else "moderately" if abs(z_score) > 0.5 else "slightly"

        # Parse feature name
        if feature_name.startswith("age30_"):
            stat = feature_name.replace("age30_", "").replace("pct", "%").replace("_", "/")
            return f"{stat} at age 30 is {magnitude} {direction} average ({value:.2f} vs {mean:.2f})"

        elif feature_name.startswith("delta_"):
            parts = feature_name.replace("delta_", "").split("_")
            stat = parts[0].replace("pct", "%")
            age = parts[1]
            change_dir = "increased" if value > 0 else "decreased"
            return f"{stat} {change_dir} by {abs(value):.2f} from age 30 to {age}"

        elif feature_name.startswith("total_"):
            stat = feature_name.replace("total_", "").replace("_change", "").replace("pct", "%")
            change_dir = "increased" if value > 0 else "decreased"
            return f"Total {stat} {change_dir} by {abs(value):.3f} from 30-33"

        elif feature_name.startswith("adaptation_velocity"):
            stat = feature_name.replace("adaptation_velocity_", "").replace("pct", "%")
            return f"Adaptation rate for {stat}: {value:.3f} per year"

        elif feature_name == "seasons_at_30":
            return f"Had {int(value)} seasons of experience by age 30"

        elif feature_name == "career_high_USG":
            return f"Career high USG% was {value:.1f}%"

        elif feature_name == "career_avg_BPM":
            return f"Career average BPM before 30 was {value:+.1f}"

        else:
            return f"{feature_name}: {value:.2f}"

    def _find_comparable_players(
        self,
        features: np.ndarray,
        n_neighbors: int = 5
    ) -> List[Dict]:
        """Find most similar players from training data."""
        # Transform features through pipeline
        imputer = self.pipeline.named_steps["imputer"]
        scaler = self.pipeline.named_steps["scaler"]

        X_imputed = imputer.transform(features.reshape(1, -1))
        X_scaled = scaler.transform(X_imputed)

        # Find neighbors
        distances, indices = self.knn.kneighbors(X_scaled)

        comparables = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.training_data.iloc[idx]
            similarity = 1 / (1 + dist)  # Convert distance to similarity

            comparables.append({
                "name": row["Player"],
                "similarity": round(similarity, 2),
                "actual_tier": int(row["adaptability_tier"]),
                "actual_label": TIER_LABELS[int(row["adaptability_tier"])],
            })

        return comparables

    def _generate_interpretation(
        self,
        player_name: str,
        predicted_label: str,
        probabilities: Dict[str, float],
        positive_factors: List[Tuple[str, float]],
        negative_factors: List[Tuple[str, float]],
        comparable_players: List[Dict],
    ) -> str:
        """Generate natural language interpretation of prediction."""
        max_prob = max(probabilities.values())
        prob_str = f"{max_prob:.0%}"

        # Opening
        if predicted_label == "THRIVED":
            opening = f"The model predicts {player_name} will THRIVE into their late 30s ({prob_str} probability)."
        elif predicted_label == "SURVIVED":
            opening = f"The model predicts {player_name} will SURVIVE as a valuable rotation player ({prob_str} probability)."
        else:
            opening = f"The model predicts {player_name} will struggle to adapt (FADED tier, {prob_str} probability)."

        # Key factors
        if positive_factors:
            top_positive = positive_factors[0][0]
            factors_str = f" Key strength: {top_positive}."
        else:
            factors_str = ""

        if negative_factors:
            top_negative = negative_factors[0][0]
            concerns_str = f" Main concern: {top_negative}."
        else:
            concerns_str = ""

        # Comparable players
        if comparable_players:
            comp_names = [c["name"] for c in comparable_players[:3]]
            comp_outcomes = [c["actual_label"] for c in comparable_players[:3]]
            comp_str = f" Similar to players like {', '.join(comp_names)} (outcomes: {', '.join(comp_outcomes)})."
        else:
            comp_str = ""

        return opening + factors_str + concerns_str + comp_str

    def predict(
        self,
        player_name: str,
        features: Dict[str, float],
    ) -> PredictionResult:
        """Make a prediction with full explanations.

        Args:
            player_name: Name of the player being predicted.
            features: Dictionary of feature values.

        Returns:
            PredictionResult with prediction and explanations.
        """
        # Convert to DataFrame row
        feature_series = pd.Series(features)

        # Ensure all features present (in correct order)
        feature_array = np.array([feature_series.get(col, np.nan) for col in self.feature_columns])

        # Get prediction and probabilities
        X = feature_array.reshape(1, -1)
        predicted_tier = self.pipeline.predict(X)[0]
        probabilities = self.pipeline.predict_proba(X)[0]

        prob_dict = {
            "FADED (0)": round(probabilities[0], 3),
            "SURVIVED (1)": round(probabilities[1], 3),
            "THRIVED (2)": round(probabilities[2], 3),
        }

        confidence = self._get_confidence_level(probabilities)

        # Get feature contributions
        positive_factors, negative_factors = self._get_feature_contributions(feature_series)

        # Find comparable players
        comparable_players = self._find_comparable_players(feature_array)

        # Generate interpretation
        interpretation = self._generate_interpretation(
            player_name,
            TIER_LABELS[predicted_tier],
            prob_dict,
            positive_factors,
            negative_factors,
            comparable_players,
        )

        return PredictionResult(
            player_name=player_name,
            predicted_tier=predicted_tier,
            predicted_label=TIER_LABELS[predicted_tier],
            probabilities=prob_dict,
            confidence=confidence,
            top_positive_factors=positive_factors,
            top_negative_factors=negative_factors,
            comparable_players=comparable_players,
            interpretation=interpretation,
        )

    def predict_from_dataframe(
        self,
        player_name: str,
        df: pd.DataFrame,
    ) -> PredictionResult:
        """Make prediction from a DataFrame row.

        Args:
            player_name: Name of the player.
            df: DataFrame containing the player's feature data.

        Returns:
            PredictionResult with prediction and explanations.
        """
        if player_name not in df["Player"].values:
            raise ValueError(f"Player '{player_name}' not found in data")

        row = df[df["Player"] == player_name].iloc[0]
        features = {col: row[col] for col in self.feature_columns}

        return self.predict(player_name, features)


def format_prediction_report(result: PredictionResult) -> str:
    """Format prediction result as a readable report."""
    lines = [
        "=" * 60,
        f"ADAPTABILITY PREDICTION: {result.player_name}",
        "=" * 60,
        "",
        f"Prediction: {result.predicted_label} (Tier {result.predicted_tier})",
        f"Confidence: {result.confidence}",
        "",
        "Probabilities:",
    ]

    for label, prob in result.probabilities.items():
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(f"  {label}: {bar} {prob:.1%}")

    lines.extend([
        "",
        "Top Positive Factors:",
    ])
    for desc, val in result.top_positive_factors[:3]:
        lines.append(f"  + {desc}")

    if result.top_negative_factors:
        lines.extend([
            "",
            "Top Concerns:",
        ])
        for desc, val in result.top_negative_factors[:3]:
            lines.append(f"  - {desc}")

    lines.extend([
        "",
        "Comparable Players:",
    ])
    for comp in result.comparable_players[:4]:
        lines.append(f"  - {comp['name']} (similarity: {comp['similarity']:.0%}, outcome: {comp['actual_label']})")

    lines.extend([
        "",
        "Interpretation:",
        result.interpretation,
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # Test the predictor
    print("Loading predictor...")
    predictor = AdaptabilityPredictor()

    print("\nTesting on training data players...")
    training_data = load_training_data()

    # Test on a few known players
    test_players = ["Tim Duncan", "Dirk Nowitzki", "Allen Iverson", "Steve Nash", "Ray Allen"]

    for player in test_players:
        if player in training_data["Player"].values:
            try:
                result = predictor.predict_from_dataframe(player, training_data)
                print(format_prediction_report(result))
                print()
            except Exception as e:
                print(f"Error predicting {player}: {e}")
