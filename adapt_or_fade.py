"""Adapt or Fade model training and prediction utilities."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

DATA_URL = os.getenv(
    "PLAYER_STATS_URL",
    "https://raw.githubusercontent.com/Adaptability-Model-NBA/Adaptability-Model-NBA/main/player_stats.csv",
)

BASE_STATS = ["3PAr", "AST%", "USG%", "TS%", "WS/48", "BPM", "MP"]


@dataclass
class ModelArtifacts:
    model: Pipeline
    feature_columns: List[str]
    report: str
    accuracy: float


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load the player stats data from a GitHub raw URL."""
    return pd.read_csv(url)


def _dedupe_season_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer TOT rows when players have multiple team entries per season."""
    if "Tm" not in df.columns:
        return df

    def _pick_tot(group: pd.DataFrame) -> pd.DataFrame:
        tot_rows = group[group["Tm"] == "TOT"]
        return tot_rows.head(1) if not tot_rows.empty else group.head(1)

    return (
        df.groupby(["Player", "Year"], as_index=False, group_keys=False)
        .apply(_pick_tot)
        .reset_index(drop=True)
    )


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to seasons with MP >= 500 and keep relevant numeric columns."""
    df = _dedupe_season_rows(df)
    df = _coerce_numeric(df, ["Age", "Year", "MP", *BASE_STATS])
    df = df[df["MP"] >= 500].copy()
    return df


def build_feature_row(player_df: pd.DataFrame) -> Tuple[Dict[str, float], int]:
    """Build features and label for a single player."""
    age_28 = player_df[player_df["Age"] == 28]
    if age_28.empty:
        raise ValueError("Player has no age-28 season")

    age_28_row = age_28.sort_values("Year").iloc[0]
    features: Dict[str, float] = {}

    for stat in BASE_STATS:
        features[f"{stat}_age28"] = age_28_row.get(stat, np.nan)

    for target_age in [29, 30, 31]:
        age_row = player_df[player_df["Age"] == target_age]
        if age_row.empty:
            for stat in BASE_STATS:
                features[f"{stat}_delta_age{target_age}"] = np.nan
            continue

        age_row = age_row.sort_values("Year").iloc[0]
        for stat in BASE_STATS:
            features[f"{stat}_delta_age{target_age}"] = (
                age_row.get(stat, np.nan) - age_28_row.get(stat, np.nan)
            )

    played_to_35 = int((player_df["Age"] >= 35).any())
    return features, played_to_35


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create model-ready features, labels, and player names."""
    df = filter_data(df)

    age_28_seasons = df[df["Age"] == 28]
    age_28_seasons = age_28_seasons[age_28_seasons["Year"].between(2005, 2018)]
    eligible_players = age_28_seasons["Player"].unique()

    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []
    players: List[str] = []

    for player in eligible_players:
        player_df = df[df["Player"] == player]
        try:
            features, label = build_feature_row(player_df)
        except ValueError:
            continue
        feature_rows.append(features)
        labels.append(label)
        players.append(player)

    features_df = pd.DataFrame(feature_rows)
    label_series = pd.Series(labels, name="played_to_35")
    player_series = pd.Series(players, name="Player")
    return features_df, label_series, player_series


def train_model(features: pd.DataFrame, labels: pd.Series) -> ModelArtifacts:
    """Train a RandomForest classifier and return artifacts."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    numeric_features = features.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return ModelArtifacts(
        model=pipeline,
        feature_columns=numeric_features,
        report=report,
        accuracy=accuracy,
    )


def save_model(artifacts: ModelArtifacts, path: str = "model.pkl") -> None:
    """Persist model artifacts to disk."""
    payload = {
        "model": artifacts.model,
        "feature_columns": artifacts.feature_columns,
    }
    joblib.dump(payload, path)


def load_model(path: str = "model.pkl") -> Tuple[Pipeline, List[str]]:
    """Load persisted model artifacts."""
    payload = joblib.load(path)
    return payload["model"], payload["feature_columns"]


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "Low"
    if probability >= 0.4:
        return "Medium"
    return "High"


def _signal_direction(delta: float) -> str:
    if np.isnan(delta):
        return "↔ stable"
    if delta > 0:
        return "↑ spacing increase"
    if delta < 0:
        return "↓ efficient role shift"
    return "↔ stable"


def predict_longevity(
    player_dict: Dict[str, Dict[str, float]],
    model: Pipeline,
    feature_columns: List[str],
) -> Dict[str, object]:
    """Predict longevity for a single player given age-28 to age-31 stats."""
    age_28 = player_dict.get("age_28", {})
    features: Dict[str, float] = {}
    for stat in BASE_STATS:
        features[f"{stat}_age28"] = age_28.get(stat, np.nan)

    for target_age in [29, 30, 31]:
        age_stats = player_dict.get(f"age_{target_age}", {})
        for stat in BASE_STATS:
            features[f"{stat}_delta_age{target_age}"] = (
                age_stats.get(stat, np.nan) - age_28.get(stat, np.nan)
            )

    feature_frame = pd.DataFrame([features])[feature_columns]
    probability = float(model.predict_proba(feature_frame)[0][1])

    signals = {
        "3PAr": _signal_direction(features["3PAr_delta_age29"]),
        "AST%": _signal_direction(features["AST%_delta_age29"]),
        "USG%": _signal_direction(features["USG%_delta_age29"]),
    }

    return {
        "Prediction": probability >= 0.5,
        "Probability": round(probability, 2),
        "Risk_Level": _risk_level(probability),
        "Key_Adaptability_Signals": signals,
        "GM_Advice": "Retain on short-term. Bench veteran or glue-guy role.",
        "Comparables": ["Iguodala", "PJ Tucker"],
    }


def feature_importances(model: Pipeline, feature_columns: List[str]) -> pd.DataFrame:
    """Return feature importances for the trained model."""
    estimator = model.named_steps["model"]
    importances = estimator.feature_importances_
    return (
        pd.DataFrame({"feature": feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_training(data_url: str = DATA_URL) -> ModelArtifacts:
    df = load_data(data_url)
    features, labels, _ = build_dataset(df)
    artifacts = train_model(features, labels)

    print("Accuracy:", artifacts.accuracy)
    print("Classification Report:\n", artifacts.report)
    print("Top Features:\n", feature_importances(artifacts.model, artifacts.feature_columns).head(10))

    save_model(artifacts)
    return artifacts


def run_batch_predictions(data_url: str = DATA_URL, output_path: str = "batch_predictions.csv") -> None:
    df = load_data(data_url)
    features, labels, players = build_dataset(df)
    model, feature_columns = load_model()

    probabilities = model.predict_proba(features[feature_columns])[:, 1]
    results = pd.DataFrame(
        {
            "Player": players,
            "Probability": probabilities,
            "Prediction": probabilities >= 0.5,
        }
    )
    results.to_csv(output_path, index=False)
    print(f"Saved batch predictions to {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train and run Adapt or Fade model")
    parser.add_argument("--data-url", default=DATA_URL, help="GitHub raw CSV URL")
    parser.add_argument("--train", action="store_true", help="Train model and save to model.pkl")
    parser.add_argument("--batch", action="store_true", help="Run batch predictions for ages 29-32")
    parser.add_argument("--predict", type=str, help="JSON string of player stats for prediction")

    args = parser.parse_args()

    if args.train:
        run_training(args.data_url)
    if args.batch:
        run_batch_predictions(args.data_url)
    if args.predict:
        model, feature_columns = load_model()
        player_dict = json.loads(args.predict)
        output = predict_longevity(player_dict, model, feature_columns)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
