# Adapt or Fade: NBA Longevity Model

This repo contains a lightweight pipeline to build and use the **"Adapt or Fade"** model, which predicts whether NBA players remain active at age 35+ based on how their game evolves after age 28.

## Quick Start

### 1) Install dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### 2) Train the model

```bash
python adapt_or_fade.py --train
```

This will:
- Load `player_stats.csv` from the repo's raw GitHub URL (or `PLAYER_STATS_URL`).
- Filter to seasons with `MP >= 500`.
- Build features based on age-28 stats and deltas at ages 29–31.
- Train a RandomForest model and save `model.pkl`.

### 3) Run a single prediction

```bash
python adapt_or_fade.py --predict '{
  "age_28": {"3PAr": 0.35, "AST%": 18.2, "USG%": 21.1, "TS%": 0.57, "WS/48": 0.12, "BPM": 1.8, "MP": 2500},
  "age_29": {"3PAr": 0.40, "AST%": 19.5, "USG%": 20.3, "TS%": 0.58, "WS/48": 0.11, "BPM": 1.5, "MP": 2400},
  "age_30": {"3PAr": 0.42, "AST%": 20.1, "USG%": 19.7, "TS%": 0.59, "WS/48": 0.10, "BPM": 1.2, "MP": 2300},
  "age_31": {"3PAr": 0.45, "AST%": 21.0, "USG%": 18.9, "TS%": 0.60, "WS/48": 0.09, "BPM": 0.8, "MP": 2200}
}'
```

You will receive a JSON response with:
- Prediction (True/False)
- Probability
- Risk level
- Key adaptability signals
- GM advice
- Comparables

### 4) Batch predictions

```bash
python batch_predict.py --output batch_predictions.csv
```

This loads the saved `model.pkl` and outputs a CSV with `Player`, `Probability`, and `Prediction`.

## Data Source Override

To load a different stats file, set `PLAYER_STATS_URL` or pass `--data-url`:

```bash
PLAYER_STATS_URL="https://raw.githubusercontent.com/your-org/your-repo/main/player_stats.csv" \
  python adapt_or_fade.py --train
```

```bash
python adapt_or_fade.py --train --data-url "https://raw.githubusercontent.com/your-org/your-repo/main/player_stats.csv"
```

## Notes

- The training cohort is limited to players whose **age-28 season** is between **2005 and 2018**.
- Only seasons with **MP >= 500** are included.
