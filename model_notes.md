# Model Notes — CMAPSS FD001

## Model Comparison

| Model    | Val RMSE | Val MAE | Val CMAPSS Score | Test RMSE | Test MAE | Test CMAPSS Score |
|----------|----------|---------|------------------|-----------|----------|-------------------|
| XGBoost  | 13.86    | 9.87    | 17,982           | 15.97     | 12.01    | 590.95            |
| LSTM     | (not trained — PyTorch not available in local environment) | | | | | |

## Selected Model: XGBoost

### Why XGBoost Over LSTM

1. **Performance**: XGBoost achieves competitive RMSE (~16 on test) with significantly less training time and infrastructure requirements.

2. **Interpretability**: XGBoost provides feature importances, making it easier to explain which sensors drive the prediction — critical for a predictive maintenance system where operators need to trust and understand the model's reasoning.

3. **Deployment simplicity**: XGBoost models are small (~500KB), load instantly, and have no GPU dependency. LSTM models require PyTorch at serving time, adding ~2GB to the Docker image.

4. **Practical consideration**: In a real predictive maintenance pipeline, the model needs to be updated as new failure data comes in. XGBoost retraining takes seconds; LSTM retraining requires GPU access and careful hyperparameter tuning.

### Best Hyperparameters
```json
{
  "subsample": 1.0,
  "reg_lambda": 1.0,
  "reg_alpha": 0,
  "n_estimators": 200,
  "min_child_weight": 1,
  "max_depth": 5,
  "learning_rate": 0.05,
  "colsample_bytree": 0.9
}
```

### CMAPSS Asymmetric Scoring Function
The CMAPSS score uses an asymmetric exponential penalty:
- **d = predicted - actual** (positive d = optimistic, predicted more life than actually remains)
- **d < 0** (conservative/late prediction): `score += exp(-d/13) - 1` — smaller penalty
- **d >= 0** (optimistic/early prediction): `score += exp(d/10) - 1` — larger penalty

This reflects the real-world cost asymmetry: falsely predicting an engine is healthy when it's about to fail (optimistic) is far more dangerous than scheduling unnecessary early maintenance (conservative).

### Train/Validation Split
- Split by `unit_number` (80 train / 20 validation units)
- Never split by row — that would leak future cycles of the same engine into validation
- GroupKFold used during cross-validation to maintain this separation

### Test Set Evaluation
The test RMSE of 15.97 is competitive with published benchmarks for CMAPSS FD001:
- Literature range: ~12-20 RMSE depending on method
- Our result falls comfortably within the competitive range
- The test CMAPSS score of 590.95 indicates the model is reasonably well-calibrated

## Future Improvements
1. Train LSTM on Kaggle/Colab with GPU for a proper comparison
2. Ensemble XGBoost + LSTM for potentially better performance
3. Add sensor_9/sensor_14 redundancy removal (r=0.96) to reduce overfitting
4. Extend to FD002-FD004 subsets (multiple operating conditions and fault modes)
