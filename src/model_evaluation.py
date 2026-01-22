"""Model evaluation and visualization for NBA Adaptability Model.

This module provides functions to evaluate model performance and generate
visualizations for analysis and reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from .model_training import load_model, TIER_LABELS
from .utils import (
    MODELS_DIR,
    OUTPUTS_DIR,
    VIZ_DIR,
    ensure_directories,
    get_feature_columns,
    load_train_test_split,
)


# Set style for all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)
    labels = ["FADED", "SURVIVED", "THRIVED"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix - Adaptability Predictions", fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importances.

    Args:
        importance_df: DataFrame with 'feature' and 'importance_pct' columns.
        top_n: Number of top features to show.
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = importance_df.head(top_n).copy()
    top_features = top_features.iloc[::-1]  # Reverse for horizontal bar

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]

    bars = ax.barh(
        top_features["feature"],
        top_features["importance_pct"],
        color=colors,
        edgecolor="navy",
        linewidth=0.5,
    )

    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)

    # Add value labels
    for bar, val in zip(bars, top_features["importance_pct"]):
        ax.text(
            val + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ROC curves for each class (one-vs-rest).

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities (n_samples x n_classes).
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["FADED", "SURVIVED", "THRIVED"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        # Binary labels for this class
        y_binary = (y_true == i).astype(int)

        if y_binary.sum() > 0:  # Only plot if class exists in test set
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
            auc = roc_auc_score(y_binary, y_proba[:, i])

            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves by Class (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_class_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot class distribution in train and test sets.

    Args:
        y_train: Training labels.
        y_test: Test labels.
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["FADED", "SURVIVED", "THRIVED"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for ax, y, title in zip(axes, [y_train, y_test], ["Training Set", "Test Set"]):
        counts = [np.sum(y == i) for i in range(3)]
        bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{title} (n={len(y)})", fontsize=14)

        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

    plt.suptitle("Class Distribution", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_model_report(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metadata: Dict,
    save_path: Optional[Path] = None,
) -> str:
    """Generate markdown report of model performance.

    Args:
        test_df: Test DataFrame with Player names and labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        metadata: Training metadata dictionary.
        save_path: Path to save report (optional).

    Returns:
        Markdown report string.
    """
    y_true = test_df["adaptability_tier"].values

    report_lines = [
        "# NBA Adaptability Model - Performance Report",
        "",
        "## Model Summary",
        "",
        f"- **Training samples**: {metadata.get('n_train', 'N/A')}",
        f"- **Test samples**: {metadata.get('n_test', 'N/A')}",
        f"- **Features**: {len(metadata.get('feature_columns', []))}",
        "",
        "## Cross-Validation Results",
        "",
    ]

    if "baseline_cv" in metadata:
        cv = metadata["baseline_cv"]
        report_lines.extend([
            f"- **Baseline CV F1**: {cv.get('cv_mean', 0):.3f} (+/- {cv.get('cv_std', 0):.3f})",
        ])

    if "tuning_results" in metadata:
        tune = metadata["tuning_results"]
        report_lines.extend([
            f"- **Tuned CV F1**: {tune.get('best_score', 0):.3f}",
            "",
            "### Best Hyperparameters",
            "",
        ])
        for param, value in tune.get("best_params", {}).items():
            clean_param = param.replace("classifier__", "")
            report_lines.append(f"- `{clean_param}`: {value}")

    report_lines.extend([
        "",
        "## Test Set Performance",
        "",
    ])

    if "test_results" in metadata:
        test = metadata["test_results"]
        report_lines.extend([
            f"- **Accuracy**: {test.get('accuracy', 0):.1%}",
            f"- **F1 (weighted)**: {test.get('f1_weighted', 0):.3f}",
            f"- **F1 (macro)**: {test.get('f1_macro', 0):.3f}",
            "",
            "### Per-Class Metrics",
            "",
            "| Class | Precision | Recall | F1 Score |",
            "|-------|-----------|--------|----------|",
        ])

        for tier_name in ["FADED", "SURVIVED", "THRIVED"]:
            if tier_name in test.get("per_class", {}):
                m = test["per_class"][tier_name]
                report_lines.append(
                    f"| {tier_name} | {m.get('precision', 0):.2f} | {m.get('recall', 0):.2f} | {m.get('f1', 0):.2f} |"
                )

    # Add prediction examples
    report_lines.extend([
        "",
        "## Test Set Predictions",
        "",
        "### Correct Predictions",
        "",
    ])

    correct_mask = y_pred == y_true
    correct_df = test_df[correct_mask].copy()
    correct_df["predicted"] = y_pred[correct_mask]

    for tier in [2, 1, 0]:
        tier_df = correct_df[correct_df["adaptability_tier"] == tier]
        if len(tier_df) > 0:
            players = tier_df["Player"].tolist()[:5]
            report_lines.append(f"- **{TIER_LABELS[tier]}**: {', '.join(players)}")

    report_lines.extend([
        "",
        "### Misclassifications",
        "",
    ])

    incorrect_mask = y_pred != y_true
    if incorrect_mask.sum() > 0:
        incorrect_df = test_df[incorrect_mask].copy()
        incorrect_df["predicted"] = y_pred[incorrect_mask]

        for _, row in incorrect_df.iterrows():
            actual = TIER_LABELS[int(row["adaptability_tier"])]
            pred = TIER_LABELS[int(row["predicted"])]
            report_lines.append(f"- {row['Player']}: Actual={actual}, Predicted={pred}")

    report_str = "\n".join(report_lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report_str)

    return report_str


def run_full_evaluation():
    """Run complete evaluation and generate all outputs."""
    ensure_directories()

    # Create subdirectories
    model_perf_dir = VIZ_DIR / "model_performance"
    feature_dir = VIZ_DIR / "feature_importance"
    model_perf_dir.mkdir(exist_ok=True)
    feature_dir.mkdir(exist_ok=True)

    print("Loading model and data...")
    pipeline, feature_cols, metadata = load_model()
    train_df, test_df = load_train_test_split()

    X_test = test_df[feature_cols]
    y_test = test_df["adaptability_tier"].values

    # Get predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    print("\nGenerating visualizations...")

    # 1. Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=model_perf_dir / "confusion_matrix.png"
    )
    print("  - Saved confusion_matrix.png")

    # 2. Feature importance
    importance_df = pd.read_csv(MODELS_DIR / "feature_importances.csv")
    plot_feature_importance(
        importance_df,
        top_n=15,
        save_path=feature_dir / "top_features_bar.png"
    )
    print("  - Saved top_features_bar.png")

    # 3. ROC curves
    plot_roc_curves(
        y_test, y_proba,
        save_path=model_perf_dir / "roc_curves.png"
    )
    print("  - Saved roc_curves.png")

    # 4. Class distribution
    y_train = train_df["adaptability_tier"].values
    plot_class_distribution(
        y_train, y_test,
        save_path=model_perf_dir / "class_distribution.png"
    )
    print("  - Saved class_distribution.png")

    # 5. Generate report
    print("\nGenerating model report...")
    report = generate_model_report(
        test_df, y_pred, y_proba, metadata,
        save_path=OUTPUTS_DIR / "model_report.md"
    )
    print("  - Saved model_report.md")

    plt.close("all")

    print("\nEvaluation complete!")
    return report


if __name__ == "__main__":
    run_full_evaluation()
