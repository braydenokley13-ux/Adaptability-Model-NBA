"""
NBA Adaptability Predictor - Interactive Web Interface
Run with: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_training_data, MODELS_DIR, VIZ_DIR, PROCESSED_DIR, ensure_directories
from src.prediction_engine import PredictionResult

# Page config
st.set_page_config(
    page_title="NBA Adaptability Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .tier-thrived {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .tier-survived {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .tier-faded {
        background: linear-gradient(135deg, #e17055, #d63031);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1e3c72;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .case-study-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .prob-bar {
        height: 25px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def check_and_train_model():
    """Check if model exists, train if not. Returns True if ready."""
    model_path = MODELS_DIR / "adaptability_model.pkl"
    features_path = MODELS_DIR / "feature_columns.pkl"
    training_data_path = PROCESSED_DIR / "training_data.csv"

    # Check if all required files exist
    if model_path.exists() and features_path.exists() and training_data_path.exists():
        return True

    # Need to train - show setup UI
    st.markdown("## 🔧 First Time Setup")
    st.info("The model needs to be trained before first use. This only happens once and takes about 30-60 seconds.")

    if st.button("🚀 Start Setup", type="primary"):
        with st.spinner("Step 1/3: Building training data..."):
            # Import and run data pipeline
            from src.data_pipeline import load_and_clean_data, identify_eligible_players
            from src.feature_engineering import build_training_dataset
            from src.utils import create_train_test_split, save_train_test_split

            ensure_directories()

            # Load and process data
            df = load_and_clean_data()
            eligible = identify_eligible_players(df)

            # Build features
            features_df, labels, players = build_training_dataset(df, eligible)

            # Save training data
            full_df = features_df.copy()
            full_df["Player"] = players.values
            full_df["adaptability_tier"] = labels.values
            full_df.to_csv(training_data_path, index=False)

            # Create train/test split
            train_df, test_df = create_train_test_split(full_df)
            save_train_test_split(train_df, test_df)

        with st.spinner("Step 2/3: Training model (this takes ~30 seconds)..."):
            from src.model_training import train_baseline_model, tune_hyperparameters, save_model, get_feature_importances
            from src.utils import get_feature_columns, load_train_test_split

            train_df, test_df = load_train_test_split()
            feature_cols = get_feature_columns(train_df)
            X_train = train_df[feature_cols]
            y_train = train_df["adaptability_tier"]

            # Train with tuning
            best_model, tune_results = tune_hyperparameters(X_train, y_train, n_iter=20)

            # Get feature importances
            importance_df = get_feature_importances(best_model, feature_cols)
            importance_df.to_csv(MODELS_DIR / "feature_importances.csv", index=False)

            # Save model
            metadata = {
                "tuning_results": {
                    "best_params": tune_results["best_params"],
                    "best_score": tune_results["best_score"],
                },
                "feature_columns": feature_cols,
                "n_train": len(X_train),
            }
            save_model(best_model, feature_cols, metadata)

        with st.spinner("Step 3/3: Finalizing..."):
            import time
            time.sleep(1)

        st.success("✅ Setup complete! Please click the button below to start the app.")
        if st.button("🏀 Launch App"):
            st.rerun()
        return False

    return False


def is_model_ready():
    """Quick check if model files exist."""
    model_path = MODELS_DIR / "adaptability_model.pkl"
    features_path = MODELS_DIR / "feature_columns.pkl"
    training_data_path = PROCESSED_DIR / "training_data.csv"
    return model_path.exists() and features_path.exists() and training_data_path.exists()


@st.cache_resource
def load_predictor():
    """Load the prediction model (cached)."""
    from src.prediction_engine import AdaptabilityPredictor
    return AdaptabilityPredictor()


@st.cache_data
def load_data():
    """Load training data (cached)."""
    return load_training_data()


@st.cache_data
def load_feature_importances():
    """Load feature importances (cached)."""
    return pd.read_csv(MODELS_DIR / "feature_importances.csv")


def render_probability_bars(probabilities):
    """Render probability bars for each tier."""
    colors = {"FADED (0)": "#e17055", "SURVIVED (1)": "#fdcb6e", "THRIVED (2)": "#00b894"}

    for label, prob in probabilities.items():
        tier_name = label.split(" ")[0]
        color = colors.get(label, "#667eea")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"""<div style="background: linear-gradient(90deg, {color} {prob*100}%, #e0e0e0 {prob*100}%);
                height: 30px; border-radius: 5px; display: flex; align-items: center; padding-left: 10px;">
                <span style="color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{tier_name}</span>
                </div>""",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(f"**{prob:.1%}**")


def render_prediction_result(result: PredictionResult):
    """Render a prediction result with styling."""

    # Tier badge
    tier_styles = {
        "THRIVED": ("tier-thrived", "🌟"),
        "SURVIVED": ("tier-survived", "✅"),
        "FADED": ("tier-faded", "📉")
    }
    style_class, emoji = tier_styles.get(result.predicted_label, ("", ""))

    st.markdown(f"""
    <div class="{style_class}">
        <h2>{emoji} {result.predicted_label}</h2>
        <p>Confidence: {result.confidence}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Probability Distribution")
    render_probability_bars(result.probabilities)

    # Factors
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Positive Factors")
        for desc, val in result.top_positive_factors[:4]:
            st.success(f"**+** {desc}")

    with col2:
        st.markdown("### ⚠️ Concerns")
        if result.top_negative_factors:
            for desc, val in result.top_negative_factors[:4]:
                st.warning(f"**-** {desc}")
        else:
            st.info("No major concerns identified")

    # Comparable players
    st.markdown("### 👥 Similar Players")
    comp_cols = st.columns(len(result.comparable_players[:4]))
    for col, comp in zip(comp_cols, result.comparable_players[:4]):
        with col:
            tier_emoji = {"THRIVED": "🌟", "SURVIVED": "✅", "FADED": "📉"}.get(comp["actual_label"], "")
            st.metric(
                label=comp["name"],
                value=f"{comp['similarity']:.0%}",
                delta=f"{tier_emoji} {comp['actual_label']}"
            )

    # Interpretation
    st.markdown("### 📝 Analysis")
    st.info(result.interpretation)


# =============================================================================
# PAGE: HOME
# =============================================================================
def page_home():
    st.markdown('<h1 class="main-header">🏀 NBA Adaptability Predictor</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 1rem; font-size: 1.2rem; color: #555;">
        Predicting which NBA players will <strong>THRIVE</strong>, <strong>SURVIVE</strong>, or <strong>FADE</strong> after age 35
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy", "78.8%", "Test Set")
    with col2:
        st.metric("Players Analyzed", "162", "2005-2012 cohort")
    with col3:
        st.metric("Features Used", "42", "Stats + Trends")
    with col4:
        st.metric("F1 Score", "0.811", "Weighted")

    st.markdown("---")

    # The three tiers
    st.markdown("## The Three Tiers of Aging")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="tier-thrived">
            <h3>🌟 THRIVED</h3>
            <p><strong>5.6% of players</strong></p>
            <p>High-impact starter at 35+</p>
            <p><em>Tim Duncan, Steve Nash, Dirk Nowitzki</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="tier-survived">
            <h3>✅ SURVIVED</h3>
            <p><strong>9.3% of players</strong></p>
            <p>Valuable rotation player</p>
            <p><em>Vince Carter, Jamal Crawford</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="tier-faded">
            <h3>📉 FADED</h3>
            <p><strong>85.2% of players</strong></p>
            <p>Out of league or minimal impact</p>
            <p><em>Allen Iverson, Gilbert Arenas</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # How it works
    st.markdown("## How It Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📊 Input Features
        - **Baseline stats at age 30**: PER, BPM, WS/48, Usage%, etc.
        - **Changes from 30→33**: How their game evolved
        - **Career context**: Experience, peak performance
        - **Adaptation trends**: Speed of role reduction
        """)

    with col2:
        st.markdown("""
        ### 🤖 Model Details
        - **Algorithm**: Random Forest Classifier
        - **Training data**: 162 players (2005-2012)
        - **Class balancing**: Weighted for imbalanced tiers
        - **Validation**: 5-fold cross-validation
        """)

    st.markdown("---")
    st.markdown("### 👈 Use the sidebar to navigate to Predictions, Model Insights, or Case Studies")


# =============================================================================
# PAGE: PREDICTIONS
# =============================================================================
def page_predictions():
    st.markdown("# 🔮 Player Predictions")

    predictor = load_predictor()
    training_data = load_data()

    tab1, tab2 = st.tabs(["📋 Select Existing Player", "✏️ Input Custom Stats"])

    # Tab 1: Select existing player
    with tab1:
        st.markdown("### Select a player from the training data")

        # Sort players alphabetically
        players = sorted(training_data["Player"].unique())

        selected_player = st.selectbox(
            "Choose a player:",
            players,
            index=players.index("Tim Duncan") if "Tim Duncan" in players else 0
        )

        if st.button("🔮 Predict", key="predict_existing"):
            with st.spinner("Analyzing..."):
                result = predictor.predict_from_dataframe(selected_player, training_data)

                # Show actual outcome
                actual_tier = training_data[training_data["Player"] == selected_player]["adaptability_tier"].iloc[0]
                actual_label = {0: "FADED", 1: "SURVIVED", 2: "THRIVED"}[actual_tier]

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"## {selected_player}")
                with col2:
                    if result.predicted_label == actual_label:
                        st.success(f"✅ Actual: {actual_label}")
                    else:
                        st.error(f"❌ Actual: {actual_label}")

                render_prediction_result(result)

    # Tab 2: Custom input
    with tab2:
        st.markdown("### Enter custom player statistics")
        st.markdown("*Fill in age-30 baseline stats and optionally add change metrics*")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Age 30 Baseline Stats")

            age30_PER = st.slider("PER (Player Efficiency)", 5.0, 35.0, 15.0, 0.5)
            age30_BPM = st.slider("BPM (Box Plus/Minus)", -6.0, 12.0, 0.0, 0.1)
            age30_WS48 = st.slider("WS/48 (Win Shares per 48)", -0.05, 0.30, 0.10, 0.01)
            age30_USG = st.slider("USG% (Usage Rate)", 10.0, 40.0, 20.0, 0.5)
            age30_AST = st.slider("AST% (Assist Percentage)", 2.0, 55.0, 15.0, 0.5)
            age30_3PAr = st.slider("3PAr (3-Point Attempt Rate)", 0.0, 0.80, 0.25, 0.01)
            age30_TS = st.slider("TS% (True Shooting)", 0.40, 0.70, 0.55, 0.01)
            age30_MP = st.slider("Minutes Played", 500, 3500, 2000, 50)

        with col2:
            st.markdown("#### Changes (Age 30 → 33)")
            st.markdown("*Leave at 0 if unknown*")

            delta_USG = st.slider("USG% Change", -15.0, 10.0, 0.0, 0.5)
            delta_AST = st.slider("AST% Change", -15.0, 15.0, 0.0, 0.5)
            delta_3PAr = st.slider("3PAr Change", -0.20, 0.30, 0.0, 0.01)
            delta_TS = st.slider("TS% Change", -0.10, 0.10, 0.0, 0.01)
            delta_BPM = st.slider("BPM Change", -8.0, 5.0, 0.0, 0.1)
            delta_MP = st.slider("Minutes Change", -2000, 500, 0, 50)

            st.markdown("#### Career Context")
            seasons_at_30 = st.slider("Seasons played by age 30", 1, 12, 8)
            career_high_USG = st.slider("Career high USG%", 10.0, 45.0, 25.0, 0.5)

        if st.button("🔮 Predict Custom Player", key="predict_custom"):
            # Build feature dict
            features = {
                "age30_USGpct": age30_USG,
                "age30_ASTpct": age30_AST,
                "age30_3PAr": age30_3PAr,
                "age30_TSpct": age30_TS,
                "age30_BPM": age30_BPM,
                "age30_WS_48": age30_WS48,
                "age30_MP": age30_MP,
                "age30_PER": age30_PER,
                "delta_USGpct_31": delta_USG / 3,
                "delta_ASTpct_31": delta_AST / 3,
                "delta_3PAr_31": delta_3PAr / 3,
                "delta_TSpct_31": delta_TS / 3,
                "delta_BPM_31": delta_BPM / 3,
                "delta_WS_48_31": 0,
                "delta_MP_31": delta_MP / 3,
                "delta_PER_31": 0,
                "delta_USGpct_32": delta_USG * 2 / 3,
                "delta_ASTpct_32": delta_AST * 2 / 3,
                "delta_3PAr_32": delta_3PAr * 2 / 3,
                "delta_TSpct_32": delta_TS * 2 / 3,
                "delta_BPM_32": delta_BPM * 2 / 3,
                "delta_WS_48_32": 0,
                "delta_MP_32": delta_MP * 2 / 3,
                "delta_PER_32": 0,
                "delta_USGpct_33": delta_USG,
                "delta_ASTpct_33": delta_AST,
                "delta_3PAr_33": delta_3PAr,
                "delta_TSpct_33": delta_TS,
                "delta_BPM_33": delta_BPM,
                "delta_WS_48_33": 0,
                "delta_MP_33": delta_MP,
                "delta_PER_33": 0,
                "seasons_at_30": seasons_at_30,
                "career_high_USG": career_high_USG,
                "career_avg_BPM": age30_BPM * 0.8,
                "position_code": 3,
                "total_3PAr_change": delta_3PAr,
                "total_USGpct_change": delta_USG,
                "total_TSpct_change": delta_TS,
                "total_ASTpct_change": delta_AST,
                "adaptation_velocity_USG": delta_USG / 3,
                "adaptation_velocity_3PAr": delta_3PAr / 3,
            }

            with st.spinner("Analyzing..."):
                result = predictor.predict("Custom Player", features)
                st.markdown("## Custom Player Prediction")
                render_prediction_result(result)


# =============================================================================
# PAGE: MODEL INSIGHTS
# =============================================================================
def page_insights():
    st.markdown("# 📈 Model Insights")

    importance_df = load_feature_importances()
    training_data = load_data()

    # Key findings
    st.markdown("## 🔑 Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="insight-card">
            <h3>💡 "Be Great at 30"</h3>
            <p>The #1 predictor of success at 35+ is baseline performance at 30.</p>
            <p><strong>THRIVED avg PER: 24.4</strong><br>FADED avg PER: 13.2</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card">
            <h3>🎯 The Three-Point Trap</h3>
            <p>Increasing 3PAr actually correlates with <strong>FADING</strong>.</p>
            <p>Desperate range expansion is a sign of decline, not adaptation.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-card">
            <h3>🔄 The Usage Paradox</h3>
            <p>Higher usage at 30 correlates with <strong>success</strong>.</p>
            <p>THRIVED: 24.4% USG<br>FADED: 17.9% USG</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card">
            <h3>⏱️ Minutes = Trust</h3>
            <p>Playing time at 30 strongly predicts longevity.</p>
            <p>THRIVED: 2,692 min<br>FADED: 1,612 min</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature importance chart
    st.markdown("## 📊 Feature Importance")

    top_n = st.slider("Number of features to show", 10, 30, 15)

    top_features = importance_df.head(top_n).copy()
    top_features = top_features.sort_values("importance_pct", ascending=True)

    # Create horizontal bar chart
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))

    bars = ax.barh(top_features["feature"], top_features["importance_pct"], color=colors)
    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, top_features["importance_pct"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Feature categories
    st.markdown("## 📦 Feature Categories")

    col1, col2 = st.columns([1, 2])

    categories = {
        "Baseline Stats (Age 30)": 42.6,
        "Changes (Deltas)": 39.3,
        "Cumulative Trends": 11.0,
        "Career Context": 7.0
    }

    with col1:
        for cat, pct in categories.items():
            st.metric(cat, f"{pct}%")

    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#1e3c72", "#2a5298", "#667eea", "#764ba2"]
        wedges, texts, autotexts = ax.pie(
            categories.values(),
            labels=categories.keys(),
            autopct="%1.1f%%",
            colors=colors,
            explode=(0.05, 0, 0, 0),
            textprops={"fontsize": 10}
        )
        ax.set_title("What Matters Most for Predicting Adaptability", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Tier comparison
    st.markdown("## 📈 Stats by Tier")

    tier_comparison = training_data.groupby("adaptability_tier").agg({
        "age30_PER": "mean",
        "age30_BPM": "mean",
        "age30_WS_48": "mean",
        "age30_USGpct": "mean",
        "age30_MP": "mean",
        "age30_3PAr": "mean"
    }).round(2)

    tier_comparison.index = tier_comparison.index.map({0: "FADED", 1: "SURVIVED", 2: "THRIVED"})
    tier_comparison.columns = ["PER", "BPM", "WS/48", "USG%", "Minutes", "3PAr"]

    st.dataframe(tier_comparison.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)


# =============================================================================
# PAGE: CASE STUDIES
# =============================================================================
def page_case_studies():
    st.markdown("# 📖 Player Case Studies")

    predictor = load_predictor()
    training_data = load_data()

    # Case study data
    case_studies = [
        {
            "name": "Tim Duncan",
            "tier": "THRIVED",
            "emoji": "🌟",
            "summary": "The Blueprint - Gradual decline, no reinvention",
            "lesson": "Duncan didn't change who he was - he became a more efficient version. He played within the Spurs system, accepted fewer minutes, and focused on defense and positioning rather than athleticism.",
            "key_stats": "PER at 30: 26.1 | Still positive BPM at 39"
        },
        {
            "name": "Steve Nash",
            "tier": "THRIVED",
            "emoji": "🌟",
            "summary": "The Point God - Already perfect for aging",
            "lesson": "Nash's success came from being perfectly suited for the modern NBA before it even arrived. His passing-first game had no athletic component to decline.",
            "key_stats": "49.2% AST rate | Expanded 3PAr to .396"
        },
        {
            "name": "Allen Iverson",
            "tier": "FADED",
            "emoji": "📉",
            "summary": "The Model's Miss - Couldn't accept role reduction",
            "lesson": "Iverson had the skills to adapt but not the willingness. Multiple reports of refusing to come off the bench. The model catches the decline but can't predict psychology.",
            "key_stats": "USG% dropped from 35.8% to 23% | Out by 34"
        },
        {
            "name": "Vince Carter",
            "tier": "SURVIVED",
            "emoji": "✅",
            "summary": "The Transformer - Complete reinvention",
            "lesson": "Carter proves transformation is possible. He went from dunker to 3-point specialist (3PAr: .273 to .604). But transformation has limits - he became useful, not dominant.",
            "key_stats": "Played until 43 | 3PAr increased to .604"
        },
        {
            "name": "Ray Allen",
            "tier": "THRIVED",
            "emoji": "🌟",
            "summary": "The Specialist - Already had the skill that ages best",
            "lesson": "Allen had the skill (shooting) that ages best. His adaptation was reduction, not reinvention. Miami gave him the perfect role.",
            "key_stats": "3PAr increased to .569 | BPM positive through 37"
        },
        {
            "name": "Jamal Crawford",
            "tier": "SURVIVED",
            "emoji": "✅",
            "summary": "The Sixth Man - Know your role",
            "lesson": "Crawford wasn't a star, but he maximized his limited baseline by perfectly understanding his value. 3x Sixth Man of the Year proves you don't need to be great to have longevity.",
            "key_stats": "Below-avg BPM but maintained role"
        }
    ]

    for study in case_studies:
        tier_colors = {
            "THRIVED": "#00b894",
            "SURVIVED": "#fdcb6e",
            "FADED": "#e17055"
        }
        color = tier_colors.get(study["tier"], "#667eea")

        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 10px;
                    border-left: 5px solid {color}; margin: 1rem 0;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h2>{study["emoji"]} {study["name"]} <span style="color: {color};">({study["tier"]})</span></h2>
            <h4 style="color: #555;">{study["summary"]}</h4>
            <p><strong>Key Stats:</strong> {study["key_stats"]}</p>
            <p style="font-style: italic; color: #666;">"{study["lesson"]}"</p>
        </div>
        """, unsafe_allow_html=True)

        # Show prediction for this player
        with st.expander(f"See {study['name']}'s Prediction Details"):
            if study["name"] in training_data["Player"].values:
                result = predictor.predict_from_dataframe(study["name"], training_data)
                render_prediction_result(result)

    st.markdown("---")

    # Summary table
    st.markdown("## 📊 Summary Comparison")

    summary_data = {
        "Player": [s["name"] for s in case_studies],
        "Actual Outcome": [s["tier"] for s in case_studies],
        "Key Trait": [
            "Gradual decline, no reinvention",
            "Passing-first, no athletic decline",
            "Couldn't accept reduced role",
            "Complete skill transformation",
            "Already a specialist shooter",
            "Understood limited role perfectly"
        ]
    }

    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    st.markdown("---")

    # Key takeaways
    st.markdown("## 🎯 Key Takeaways")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🏆 The Duncan Rule</h4>
            <p>The best adapters don't reinvent themselves - they become more focused versions of who they already were.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <h4>⚠️ The Iverson Warning</h4>
            <p>Talent without flexibility is a death sentence. Stats can predict trajectory but not psychology.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 The Carter Ceiling</h4>
            <p>Complete transformation is possible but has limits. You can extend your career, but you can't maintain stardom through reinvention alone.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <h4>✅ The Crawford Path</h4>
            <p>Know your role. Middle-tier players who accept their limitations often outlast stars who can't.</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Check if model is ready - if not, show setup page
    if not is_model_ready():
        st.markdown('<h1 class="main-header">🏀 NBA Adaptability Predictor</h1>', unsafe_allow_html=True)
        check_and_train_model()
        return

    # Sidebar navigation
    st.sidebar.markdown("# 🏀 Navigation")

    pages = {
        "🏠 Home": page_home,
        "🔮 Predictions": page_predictions,
        "📈 Model Insights": page_insights,
        "📖 Case Studies": page_case_studies
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This model predicts NBA player longevity based on their performance evolution from ages 30-33.

    **Model**: Random Forest
    **Accuracy**: 78.8%
    **Data**: 2005-2017 seasons
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with Streamlit")

    # Render selected page
    pages[selection]()


if __name__ == "__main__":
    main()
