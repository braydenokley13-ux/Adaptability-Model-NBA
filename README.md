# NBA Adaptability Prediction Model

A machine learning model that predicts whether NBA players will **THRIVE**, **SURVIVE**, or **FADE** after age 35, based on their performance and adaptation patterns from ages 30-33.

## Overview

This project uses a Random Forest classifier trained on 162 players who turned 30 between 2005-2012, allowing us to observe their performance through age 35+.

### The Three Tiers

| Tier | Label | Criteria | Examples |
|------|-------|----------|----------|
| 2 | **THRIVED** | 3+ seasons at 35+ with high impact (WS/48 ≥ 0.100 or BPM ≥ 1.5) | Tim Duncan, Steve Nash, Dirk Nowitzki |
| 1 | **SURVIVED** | 2+ seasons at 35+ with positive contribution | Vince Carter, Jamal Crawford, Derek Fisher |
| 0 | **FADED** | Did not maintain career past 35 or minimal impact | Allen Iverson, Carmelo Anthony |

### Model Performance

- **Test Accuracy**: 78.8%
- **F1 Score (weighted)**: 0.811
- **Top Predictor**: PER at age 30 (11.2% importance)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Web Interface

```bash
streamlit run app.py
```

This opens an interactive web app where you can:
- **Select existing players** from a dropdown and see their prediction
- **Input custom stats** using sliders and get instant predictions
- **Explore model insights** including feature importance and key findings
- **Read case studies** of players like Tim Duncan, Allen Iverson, Vince Carter

### 3. Train the Model (Optional)

```bash
python -m src.model_training
```

This will:
- Load and process `Seasons_Stats.csv`
- Build 42 features from ages 30-33
- Train a Random Forest with hyperparameter tuning
- Save the model to `models/adaptability_model.pkl`

### 3. Make Predictions

```bash
# Predict a single player from training data
python predict_player.py "Tim Duncan"

# List available players
python predict_player.py --list

# Batch predict all players
python predict_player.py --batch outputs/predictions.csv
```

### Example Output

```
============================================================
ADAPTABILITY PREDICTION: Tim Duncan
============================================================

Prediction: THRIVED (Tier 2)
Confidence: MEDIUM

Probabilities:
  FADED (0): █████░░░░░░░░░░░░░░░░░░░░░░░░░ 19.2%
  SURVIVED (1): ████████░░░░░░░░░░░░░░░░░░░░░░ 27.2%
  THRIVED (2): ████████████████░░░░░░░░░░░░░░ 53.6%

Top Positive Factors:
  + PER at age 30 is significantly above average (26.10 vs 13.81)
  + WS/48 at age 30 is significantly above average (0.23 vs 0.10)

Comparable Players:
  - Manu Ginobili (similarity: 18%, outcome: THRIVED)
  - Paul Pierce (similarity: 16%, outcome: THRIVED)
```

## Key Findings

### 1. "Be Great at 30"
The #1 predictor of success at 35+ is baseline performance at 30. THRIVED players had an average PER of 24.4 at age 30 vs 13.2 for FADED players.

### 2. The Usage Paradox
Counter-intuitively, **higher usage at 30 correlates with success**. Stars have more value to redistribute when adapting.

### 3. The Three-Point Trap
Increasing 3PAr actually correlates with FADING. Desperate range expansion is a sign of decline, not adaptation.

### 4. Feature Importance Breakdown

| Category | Importance |
|----------|------------|
| Baseline Stats (Age 30) | 42.6% |
| Changes (Deltas 30→33) | 39.3% |
| Cumulative Trends | 11.0% |
| Career Context | 7.0% |

## Project Structure

```
nba-adaptability-ml/
├── src/
│   ├── data_pipeline.py        # Data loading and cleaning
│   ├── feature_engineering.py  # Feature extraction and labeling
│   ├── model_training.py       # Model training and tuning
│   ├── model_evaluation.py     # Evaluation and visualization
│   ├── prediction_engine.py    # Prediction with explanations
│   └── utils.py                # Utility functions
├── data/
│   └── processed/
│       ├── training_data.csv   # Features + labels
│       ├── train_set.csv       # 80% training
│       └── test_set.csv        # 20% testing
├── models/
│   ├── adaptability_model.pkl  # Trained model
│   ├── feature_columns.pkl     # Feature order
│   └── training_metadata.json  # Training info
├── outputs/
│   ├── model_report.md         # Performance summary
│   ├── feature_insights.md     # What the model learned
│   ├── case_studies.md         # Player deep dives
│   ├── podcast_script_draft.md # Podcast content
│   └── adaptability_rankings.csv
├── visualizations/
│   ├── model_performance/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves.png
│   │   └── class_distribution.png
│   ├── feature_importance/
│   │   └── top_features_bar.png
│   └── player_examples/
│       ├── tim_duncan_trajectory.png
│       ├── allen_iverson_trajectory.png
│       └── ...
├── app.py                      # Streamlit web interface
├── predict_player.py           # CLI prediction tool
├── Seasons_Stats.csv           # Raw data
├── requirements.txt            # Python dependencies
└── README.md
```

## Features Used (42 total)

### Baseline Stats at Age 30
- `age30_USGpct` - Usage rate
- `age30_ASTpct` - Assist percentage
- `age30_3PAr` - Three-point attempt rate
- `age30_TSpct` - True shooting percentage
- `age30_BPM` - Box plus/minus
- `age30_WS_48` - Win shares per 48 minutes
- `age30_MP` - Minutes played
- `age30_PER` - Player efficiency rating

### Change Metrics (Deltas)
- `delta_*_31` - Changes from age 30 to 31
- `delta_*_32` - Changes from age 30 to 32
- `delta_*_33` - Changes from age 30 to 33

### Career Context
- `seasons_at_30` - NBA experience by age 30
- `career_high_USG` - Peak usage rate
- `career_avg_BPM` - Career impact before 30
- `position_code` - Position (1-5)

### Cumulative Trends
- `total_*_change` - Total change 30→33
- `adaptation_velocity_*` - Rate of change per year

## Limitations

1. **Small THRIVED sample** (9 players) - Class imbalance addressed with balanced weights
2. **Data ends 2017** - Cannot evaluate recent players (LeBron, Chris Paul)
3. **No injury data** - Major factor in career decline not captured
4. **No team context** - System fit matters but isn't modeled

## For the Podcast

See `outputs/podcast_script_draft.md` for a complete episode script including:
- 45-60 minute format with 6 segments
- Behavioral economics angle (loss aversion, sunk cost)
- Player case studies (Duncan, Nash, Iverson, Carter)
- Quotable one-liners

## License

MIT

## Data Source

Basketball Reference data via Kaggle (1950-2017 NBA seasons).
