# Session 1: Data Pipeline & Feature Engineering - EDA Summary

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | 162 players |
| Total Features | 42 |
| Age-30 Season Years | 2005-2012 |
| Observation Period | Through 2017 (age 35+) |

## Target Variable Distribution

| Tier | Label | Count | Percentage |
|------|-------|-------|------------|
| 2 | THRIVED | 9 | 5.6% |
| 1 | SURVIVED | 15 | 9.3% |
| 0 | FADED | 138 | 85.2% |

### Class Imbalance Note
The heavy skew toward FADED is expected - most NBA players don't remain productive into their late 30s. We'll address this with `class_weight='balanced'` in the Random Forest model.

## Key Feature Statistics at Age 30

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| USG% | 18.5% | 5.2 | 7.9% | 35.8% |
| AST% | 13.5% | 9.0 | 1.7% | 49.2% |
| 3PAr | 0.248 | 0.218 | 0.000 | 0.843 |
| TS% | 53.0% | 4.7 | 33.8% | 62.9% |
| BPM | -0.37 | 2.64 | -5.6 | +8.1 |
| WS/48 | 0.098 | 0.053 | -0.031 | 0.232 |
| Minutes | 1732 | 737 | 504 | 3126 |

## Feature Means by Tier

| Feature | FADED | SURVIVED | THRIVED |
|---------|-------|----------|---------|
| USG% | 17.9 | 20.3 | 24.4 |
| AST% | 13.0 | 13.8 | 20.5 |
| 3PAr | 0.256 | 0.217 | 0.177 |
| TS% | 52.4% | 55.9% | 56.2% |
| BPM | -0.9 | +1.4 | +4.3 |
| WS/48 | 0.088 | 0.133 | 0.184 |
| Minutes | 1612 | 2259 | 2692 |

## Missing Data Analysis

Delta features (changes from age 30 to 31/32/33) have significant missingness:
- Age 31 deltas: ~22% missing
- Age 32 deltas: ~33% missing
- Age 33 deltas: ~50% missing

**Interpretation**: Missingness is informative! Players who don't have qualifying seasons at ages 32-33 were often already fading from the league. We'll use median imputation and let the model learn from available data.

## THRIVED Players (9 total)
- Tim Duncan
- Dirk Nowitzki
- Steve Nash
- Kevin Garnett
- Paul Pierce
- Ray Allen
- Manu Ginobili
- Ben Wallace
- Marcus Camby

## SURVIVED Players (sample of 15)
- Jason Terry
- Jamal Crawford
- Derek Fisher
- David West
- Andre Miller
- Antawn Jamison
- Antonio McDyess

## Key Insights for Model Training

### 1. Baseline Impact Matters Most
THRIVED players had significantly higher BPM (+4.3) and WS/48 (0.184) at age 30 compared to FADED players (-0.9 BPM, 0.088 WS/48). **Star power at 30 predicts longevity.**

### 2. Usage Paradox
Higher usage at age 30 correlates with success (THRIVED: 24.4% USG vs FADED: 17.9%). This contradicts the naive hypothesis that high-usage players struggle to adapt. **Stars who maintained high usage were actually the ones who thrived.**

### 3. Three-Point Rate Surprise
FADED players actually showed *higher* 3PAr increase (+0.042) vs THRIVED (+0.029). This suggests:
- THRIVED players already had complete games
- FADED players may have been desperately trying to add range
- **It's not about adding range - it's about having impact**

### 4. Minutes = Trust
THRIVED players averaged 2692 minutes at age 30 vs 1612 for FADED. **Playing time is a strong signal of team confidence.**

## Data Files Created

1. `data/processed/training_data.csv` - Full dataset (162 players, 42 features)
2. `data/processed/train_set.csv` - Training set (129 players, 80%)
3. `data/processed/test_set.csv` - Test set (33 players, 20%)

## Next Steps (Session 2)

1. Train Random Forest with `class_weight='balanced'`
2. Tune hyperparameters via cross-validation
3. Evaluate per-class performance
4. Extract feature importances
