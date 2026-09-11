"""
Prepare data artifacts for the Streamlit Dashboard.

Outputs:
  - dashboard/data/test_fd001.parquet: Clean test sensor readings with true ground-truth RUL
  - dashboard/data/xgb_predictions.json: Precomputed XGBoost baseline predictions per (unit, cycle)
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "serving", "model_artifacts")
DASHBOARD_DATA_DIR = os.path.join(PROJECT_ROOT, "dashboard", "data")

os.makedirs(DASHBOARD_DATA_DIR, exist_ok=True)

# 1. Copy clean test parquet
src_test_parquet = os.path.join(DATA_DIR, "test_fd001.parquet")
dst_test_parquet = os.path.join(DASHBOARD_DATA_DIR, "test_fd001.parquet")

df_test = pd.read_parquet(src_test_parquet)
df_test.to_parquet(dst_test_parquet, index=False)
print(f"[OK] Saved {dst_test_parquet} ({len(df_test)} rows, {df_test['unit_number'].nunique()} units)")

# 2. Precompute XGBoost predictions using validated model.json and Stage-2 features
src_features_parquet = os.path.join(DATA_DIR, "features_test.parquet")
features_test = pd.read_parquet(src_features_parquet)

scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
scaler = joblib.load(scaler_path)
feature_names = list(scaler.feature_names_in_)

booster = xgb.Booster()
booster.load_model(os.path.join(ARTIFACTS_DIR, "model.json"))

dmatrix = xgb.DMatrix(features_test[feature_names].values)
raw_predictions = booster.predict(dmatrix)

# Validate metrics on test_last
test_last = features_test.groupby("unit_number").last().reset_index()
dtest_last = xgb.DMatrix(test_last[feature_names].values)
test_last_preds = booster.predict(dtest_last)
rmse = float(np.sqrt(np.mean((test_last["RUL"].values - test_last_preds) ** 2)))
mae = float(np.mean(np.abs(test_last["RUL"].values - test_last_preds)))

print(f"[CHECK] XGBoost Test Last Evaluation: RMSE={rmse:.2f} (expect ~15.97), MAE={mae:.2f} (expect ~12.01)")
assert 15.0 <= rmse <= 17.0, f"RMSE {rmse} outside expected range [15.0, 17.0]"

# Format predictions: {unit_id: {cycle: predicted_rul}}
xgb_dict = {}
for unit_id, group in features_test.groupby("unit_number"):
    unit_str = str(int(unit_id))
    xgb_dict[unit_str] = {}
    unit_indices = group.index
    unit_preds = raw_predictions[unit_indices]
    for cycle, pred in zip(group["time_cycles"], unit_preds):
        cycle_str = str(int(cycle))
        xgb_dict[unit_str][cycle_str] = max(0.0, round(float(pred), 2))

dst_xgb_json = os.path.join(DASHBOARD_DATA_DIR, "xgb_predictions.json")
with open(dst_xgb_json, "w") as f:
    json.dump(xgb_dict, f)

print(f"[OK] Saved {dst_xgb_json} ({len(xgb_dict)} units, {sum(len(v) for v in xgb_dict.values())} data points)")
