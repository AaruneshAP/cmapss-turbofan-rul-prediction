"""
Stage 2 — Feature Engineering
===============================
Builds predictive features from CMAPSS sensor data for RUL regression.

This script:
  1. Drops zero/near-zero variance sensors identified in Stage 1
  2. Computes rolling-window features (mean, std, slope) per unit per sensor
  3. Analyzes sensor-to-sensor correlation for redundancy detection
  4. Clips training RUL at 125 cycles (standard practice)
  5. Fits StandardScaler on train, transforms both train and test
  6. Saves feature matrices and scaler for downstream use
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = DATA_DIR
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "serving", "model_artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# RUL clipping ceiling — standard CMAPSS practice.
# Rationale: In early life, sensor readings are essentially identical regardless
# of how long the engine will last. A model cannot distinguish RUL=200 from
# RUL=300 using sensor data alone. Clipping at 125 makes the regression
# target more learnable and focuses model capacity on the critical
# degradation phase.
RUL_CLIP = 125

# Rolling window sizes
WINDOW_SIZES = [5, 20]


def load_stage1_outputs():
    """Load parquet files and exploration metadata from Stage 1."""
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_fd001.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test_fd001.parquet"))

    meta_path = os.path.join(DATA_DIR, "exploration_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df, meta


def drop_low_variance_sensors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    low_var_sensors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop sensors with near-zero variance.
    
    These sensors provide no discriminating information — they read the same
    value regardless of engine health state. Keeping them adds noise and
    dimensionality without benefit.
    """
    print(f"\n--- Dropping Low-Variance Sensors ---")
    print(f"Sensors to drop: {low_var_sensors}")
    
    for sensor in low_var_sensors:
        if sensor in train_df.columns:
            print(f"  Dropping {sensor} — std={train_df[sensor].std():.6f}")
            train_df = train_df.drop(columns=[sensor])
            test_df = test_df.drop(columns=[sensor])

    print(f"Remaining columns: {len(train_df.columns)}")
    return train_df, test_df


def compute_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_sizes: list[int],
) -> pd.DataFrame:
    """
    Compute rolling-window features per unit, per sensor.
    
    For each sensor and window size, we compute:
      - Rolling mean: smoothed signal, reduces noise
      - Rolling std: captures variability/instability in the signal
      - Rolling slope: linear trend coefficient — a direct proxy for
        degradation rate. A steepening negative slope indicates
        accelerating degradation.
    
    NaN values from the window edges are forward/backward filled.
    """
    df = df.copy()
    new_cols = {}

    for sensor in sensor_cols:
        for w in window_sizes:
            prefix = f"{sensor}_w{w}"

            grouped = df.groupby("unit_number")[sensor]

            # Rolling mean
            new_cols[f"{prefix}_mean"] = grouped.transform(
                lambda x: x.rolling(window=w, min_periods=1).mean()
            )

            # Rolling std
            new_cols[f"{prefix}_std"] = grouped.transform(
                lambda x: x.rolling(window=w, min_periods=1).std().fillna(0)
            )

            # Rolling slope (linear regression coefficient)
            # For each window, fit y = a*x + b and extract 'a'
            def rolling_slope(series, window=w):
                """Compute slope of linear fit over rolling window."""
                slopes = []
                for i in range(len(series)):
                    start = max(0, i - window + 1)
                    segment = series.iloc[start : i + 1].values
                    if len(segment) < 2:
                        slopes.append(0.0)
                    else:
                        x = np.arange(len(segment))
                        # Use polyfit for speed — degree 1
                        coeffs = np.polyfit(x, segment, 1)
                        slopes.append(coeffs[0])
                return pd.Series(slopes, index=series.index)

            new_cols[f"{prefix}_slope"] = grouped.transform(rolling_slope)

    # Concatenate all new columns at once (faster than repeated assignment)
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    print(f"Rolling features added: {len(new_cols)} new columns")
    print(f"Total columns: {len(df.columns)}")
    return df


def analyze_sensor_correlations(
    df: pd.DataFrame, sensor_cols: list[str], threshold: float = 0.95
) -> list[tuple[str, str, float]]:
    """
    Identify highly correlated sensor pairs.
    
    Near-duplicate sensors (|corr| > threshold) are candidates for
    redundancy removal. We log them but don't auto-drop — the decision
    should be deliberate and documented.
    """
    corr_matrix = df[sensor_cols].corr().abs()

    high_corr_pairs = []
    for i in range(len(sensor_cols)):
        for j in range(i + 1, len(sensor_cols)):
            if corr_matrix.iloc[i, j] > threshold:
                pair = (sensor_cols[i], sensor_cols[j], float(corr_matrix.iloc[i, j]))
                high_corr_pairs.append(pair)

    print(f"\n--- High Correlation Pairs (|r| > {threshold}) ---")
    if high_corr_pairs:
        for s1, s2, r in high_corr_pairs:
            print(f"  {s1} <-> {s2}: r = {r:.4f}")
    else:
        print("  None found.")

    return high_corr_pairs


def clip_rul(df: pd.DataFrame, cap: int = RUL_CLIP) -> pd.DataFrame:
    """
    Clip RUL at a ceiling value.
    
    Standard CMAPSS practice — early-life cycles are indistinguishable from
    sensor data, so predicting exact RUL above ~125 is not meaningful.
    This makes the regression target more learnable.
    """
    df = df.copy()
    original_max = df["RUL"].max()
    df["RUL"] = df["RUL"].clip(upper=cap)
    print(f"\nRUL clipped: max {original_max} -> {cap}")
    return df


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on training data only, transform both sets.
    
    This prevents data leakage — test set statistics must not influence
    the scaling parameters.
    """
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    print(f"\nScaler fit on {len(feature_cols)} features from training data.")
    return train_df, test_df, scaler


def main():
    print("=" * 60)
    print("CMAPSS Feature Engineering — Stage 2")
    print("=" * 60)

    # 1. Load Stage 1 outputs
    train_df, test_df, meta = load_stage1_outputs()
    low_var_sensors = meta["low_variance_sensors"]

    # 2. Drop low-variance sensors
    train_df, test_df = drop_low_variance_sensors(train_df, test_df, low_var_sensors)

    # 3. Identify remaining sensor columns
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    setting_cols = [c for c in train_df.columns if c.startswith("setting_")]
    print(f"\nActive sensors: {sensor_cols}")

    # 4. Analyze correlations BEFORE adding rolling features
    high_corr_pairs = analyze_sensor_correlations(train_df, sensor_cols)

    # 5. Compute rolling features
    print("\n--- Computing Rolling Features ---")
    print("(This may take a minute for large datasets...)")
    train_df = compute_rolling_features(train_df, sensor_cols, WINDOW_SIZES)
    test_df = compute_rolling_features(test_df, sensor_cols, WINDOW_SIZES)

    # 6. Clip RUL
    train_df = clip_rul(train_df, RUL_CLIP)
    test_df = clip_rul(test_df, RUL_CLIP)

    # 7. Define feature columns (everything except identifiers and target)
    non_feature_cols = ["unit_number", "time_cycles", "RUL"]
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols[:10]}... (truncated)")

    # 8. Scale features
    train_df, test_df, scaler = scale_features(train_df, test_df, feature_cols)

    # 9. Save outputs
    train_out = os.path.join(OUTPUT_DIR, "features_train.parquet")
    test_out = os.path.join(OUTPUT_DIR, "features_test.parquet")
    scaler_out = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    feature_list_out = os.path.join(ARTIFACTS_DIR, "feature_list.json")

    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)
    joblib.dump(scaler, scaler_out)

    with open(feature_list_out, "w") as f:
        json.dump({"feature_cols": feature_cols, "sensor_cols": sensor_cols}, f, indent=2)

    print(f"\nSaved: {train_out}")
    print(f"Saved: {test_out}")
    print(f"Saved: {scaler_out}")
    print(f"Saved: {feature_list_out}")

    # 10. Save feature engineering metadata
    fe_meta = {
        "rul_clip": RUL_CLIP,
        "window_sizes": WINDOW_SIZES,
        "n_features": len(feature_cols),
        "dropped_sensors": low_var_sensors,
        "high_corr_pairs": [
            {"sensor_1": s1, "sensor_2": s2, "correlation": r}
            for s1, s2, r in high_corr_pairs
        ],
        "feature_cols": feature_cols,
    }
    fe_meta_path = os.path.join(DATA_DIR, "feature_engineering_meta.json")
    with open(fe_meta_path, "w") as f:
        json.dump(fe_meta, f, indent=2)
    print(f"Saved: {fe_meta_path}")

    print("\n[DONE] Stage 2 complete.")
    return fe_meta


if __name__ == "__main__":
    fe_meta = main()
