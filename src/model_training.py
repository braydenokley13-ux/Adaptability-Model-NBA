"""Model training for NBA Adaptability Prediction.

This module trains a Random Forest classifier to predict player adaptability
tiers (THRIVED/SURVIVED/FADED) based on age 30-33 performance features.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import (
    MODELS_DIR,
    PROCESSED_DIR,
    ensure_directories,
    get_feature_columns,
    load_train_test_split,
)


# Tier labels for display
TIER_LABELS = {0: "FADED", 1: "SURVIVED", 2: "THRIVED"}


def create_model_pipeline(
    n_estimators: int = 500,
    max_depth: int = 12,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> Pipeline:
    """Create the full model pipeline with preprocessing.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees.
        min_samples_split: Minimum samples to split an internal node.
        min_samples_leaf: Minimum samples at a leaf node.
        random_state: Random seed for reproducibility.

    Returns:
        Sklearn Pipeline with imputer, scaler, and classifier.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",  # Handle class imbalance
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ))
    ])
    return pipeline


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict]:
    """Train baseline model with default hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed.

    Returns:
        Tuple of (trained pipeline, cross-validation results dict).
    """
    pipeline = create_model_pipeline(random_state=random_state)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")

    # Fit on full training data
    pipeline.fit(X_train, y_train)

    cv_results = {
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }

    return pipeline, cv_results


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict]:
    """Tune hyperparameters using randomized search.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_iter: Number of parameter combinations to try.
        random_state: Random seed.

    Returns:
        Tuple of (best pipeline, search results dict).
    """
    base_pipeline = create_model_pipeline(random_state=random_state)

    param_distributions = {
        "classifier__n_estimators": [200, 300, 500, 700],
        "classifier__max_depth": [8, 10, 12, 15, None],
        "classifier__min_samples_split": [10, 15, 20, 30],
        "classifier__min_samples_leaf": [5, 8, 10, 15],
        "classifier__max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    search_results = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "cv_results_mean": search.cv_results_["mean_test_score"].tolist(),
    }

    return search.best_estimator_, search_results


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """Evaluate model on test set.

    Args:
        pipeline: Trained pipeline.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with evaluation metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report as string
    report = classification_report(y_test, y_pred, target_names=["FADED", "SURVIVED", "THRIVED"])

    results = {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "per_class": {
            TIER_LABELS[i]: {
                "precision": float(precision_per_class[i]) if i < len(precision_per_class) else 0,
                "recall": float(recall_per_class[i]) if i < len(recall_per_class) else 0,
                "f1": float(f1_per_class[i]) if i < len(f1_per_class) else 0,
            }
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": y_pred.tolist(),
        "probabilities": y_proba.tolist(),
    }

    return results


def get_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract feature importances from trained model.

    Args:
        pipeline: Trained pipeline with RandomForest classifier.
        feature_names: List of feature column names.

    Returns:
        DataFrame with features sorted by importance.
    """
    rf_model = pipeline.named_steps["classifier"]
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df["importance_pct"] = importance_df["importance"] * 100
    importance_df["cumulative_pct"] = importance_df["importance_pct"].cumsum()

    return importance_df.reset_index(drop=True)


def save_model(
    pipeline: Pipeline,
    feature_columns: List[str],
    metadata: Dict,
    model_name: str = "adaptability_model",
) -> Dict[str, Path]:
    """Save trained model and associated files.

    Args:
        pipeline: Trained pipeline.
        feature_columns: List of feature column names in order.
        metadata: Training metadata dictionary.
        model_name: Base name for saved files.

    Returns:
        Dictionary with paths to saved files.
    """
    ensure_directories()

    # Save model
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(pipeline, model_path)

    # Save feature columns
    features_path = MODELS_DIR / "feature_columns.pkl"
    joblib.dump(feature_columns, features_path)

    # Save metadata
    metadata["saved_at"] = datetime.now().isoformat()
    metadata_path = MODELS_DIR / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "model": model_path,
        "features": features_path,
        "metadata": metadata_path,
    }


def load_model() -> Tuple[Pipeline, List[str], Dict]:
    """Load saved model and associated files.

    Returns:
        Tuple of (pipeline, feature_columns, metadata).
    """
    model_path = MODELS_DIR / "adaptability_model.pkl"
    features_path = MODELS_DIR / "feature_columns.pkl"
    metadata_path = MODELS_DIR / "training_metadata.json"

    pipeline = joblib.load(model_path)
    feature_columns = joblib.load(features_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return pipeline, feature_columns, metadata


if __name__ == "__main__":
    print("=" * 60)
    print("NBA ADAPTABILITY MODEL - TRAINING")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    train_df, test_df = load_train_test_split()

    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df["adaptability_tier"]
    X_test = test_df[feature_cols]
    y_test = test_df["adaptability_tier"]

    print(f"   Train: {len(X_train)} samples, {len(feature_cols)} features")
    print(f"   Test: {len(X_test)} samples")

    # Train baseline
    print("\n2. Training baseline model...")
    baseline_model, baseline_cv = train_baseline_model(X_train, y_train)
    print(f"   CV F1 (weighted): {baseline_cv['cv_mean']:.3f} (+/- {baseline_cv['cv_std']:.3f})")

    # Hyperparameter tuning
    print("\n3. Tuning hyperparameters...")
    best_model, tune_results = tune_hyperparameters(X_train, y_train, n_iter=30)
    print(f"   Best CV F1: {tune_results['best_score']:.3f}")
    print(f"   Best params: {tune_results['best_params']}")

    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    eval_results = evaluate_model(best_model, X_test, y_test)
    print(f"   Accuracy: {eval_results['accuracy']:.3f}")
    print(f"   F1 (weighted): {eval_results['f1_weighted']:.3f}")
    print(f"   F1 (macro): {eval_results['f1_macro']:.3f}")

    print("\n   Per-class performance:")
    for tier_name, metrics in eval_results["per_class"].items():
        print(f"     {tier_name}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")

    print("\n   Confusion Matrix:")
    cm = np.array(eval_results["confusion_matrix"])
    print("              Pred: FADED  SURVIVED  THRIVED")
    for i, row in enumerate(cm):
        print(f"   Actual {TIER_LABELS[i]:>8}: {row[0]:6d}  {row[1]:8d}  {row[2]:7d}")

    # Feature importances
    print("\n5. Top 10 Feature Importances:")
    importance_df = get_feature_importances(best_model, feature_cols)
    for i, row in importance_df.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:30s} {row['importance_pct']:5.2f}%")

    # Save model
    print("\n6. Saving model...")
    metadata = {
        "baseline_cv": baseline_cv,
        "tuning_results": {
            "best_params": tune_results["best_params"],
            "best_score": tune_results["best_score"],
        },
        "test_results": {
            "accuracy": eval_results["accuracy"],
            "f1_weighted": eval_results["f1_weighted"],
            "f1_macro": eval_results["f1_macro"],
            "per_class": eval_results["per_class"],
        },
        "feature_columns": feature_cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    saved_paths = save_model(best_model, feature_cols, metadata)
    for name, path in saved_paths.items():
        print(f"   {name}: {path}")

    # Save feature importances
    importance_path = MODELS_DIR / "feature_importances.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"   feature_importances: {importance_path}")

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
