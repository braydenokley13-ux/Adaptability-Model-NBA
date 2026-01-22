# NBA Adaptability Model - Performance Report

## Model Summary

- **Training samples**: 129
- **Test samples**: 33
- **Features**: 42

## Cross-Validation Results

- **Baseline CV F1**: 0.816 (+/- 0.018)
- **Tuned CV F1**: 0.838

### Best Hyperparameters

- `n_estimators`: 500
- `min_samples_split`: 15
- `min_samples_leaf`: 15
- `max_features`: log2
- `max_depth`: 8

## Test Set Performance

- **Accuracy**: 78.8%
- **F1 (weighted)**: 0.811
- **F1 (macro)**: 0.581

### Per-Class Metrics

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| FADED | 0.96 | 0.82 | 0.88 |
| SURVIVED | 0.25 | 0.33 | 0.29 |
| THRIVED | 0.40 | 1.00 | 0.57 |

## Test Set Predictions

### Correct Predictions

- **THRIVED**: Ray Allen, Kevin Garnett
- **SURVIVED**: Shawn Marion
- **FADED**: Earl Boykins, Rashard Lewis, Marko Jaric, Corey Maggette, Trenton Hassell

### Misclassifications

- Antonio McDyess: Actual=SURVIVED, Predicted=FADED
- Joe Johnson: Actual=FADED, Predicted=THRIVED
- Zydrunas Ilgauskas: Actual=FADED, Predicted=THRIVED
- Kobe Bryant: Actual=FADED, Predicted=SURVIVED
- Chauncey Billups: Actual=FADED, Predicted=SURVIVED
- David West: Actual=SURVIVED, Predicted=THRIVED
- Peja Stojakovic: Actual=FADED, Predicted=SURVIVED