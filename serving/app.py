"""
Stage 4 — FastAPI Serving Layer
=================================
Serves CMAPSS RUL predictions via a REST API.

Endpoints:
  POST /predict — accepts sensor readings, returns predicted RUL
  GET  /health  — returns service health and model load status

The model and scaler are loaded once at startup for efficiency.
"""

import os
import json
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
import xgboost as xgb
import joblib
from fastapi import FastAPI, HTTPException
from schemas import PredictionRequest, PredictionResponse, HealthResponse

# ---------------------------------------------------------------------------
# Globals — loaded at startup
# ---------------------------------------------------------------------------
MODEL = None
SCALER = None
FEATURE_COLS = None
SENSOR_COLS = None
MODEL_VERSION = "xgboost-fd001-v1"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

# Rolling window settings (must match Stage 2)
WINDOW_SIZES = [5, 20]

# Sensors that were dropped in Stage 2 (low variance in FD001)
# Loaded from feature_list.json at startup
DROPPED_SENSORS = None


def load_model_artifacts():
    """Load model, scaler, and feature configuration at startup."""
    global MODEL, SCALER, FEATURE_COLS, SENSOR_COLS, DROPPED_SENSORS

    # Load XGBoost model
    model_path = os.path.join(ARTIFACTS_DIR, "model.json")
    if os.path.exists(model_path):
        MODEL = xgb.XGBRegressor()
        MODEL.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        # Fallback to pkl
        pkl_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
        MODEL = joblib.load(pkl_path)
        print(f"Loaded model from {pkl_path}")

    # Load scaler
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    SCALER = joblib.load(scaler_path)
    print(f"Loaded scaler from {scaler_path}")

    # Load feature list
    feat_path = os.path.join(ARTIFACTS_DIR, "feature_list.json")
    with open(feat_path) as f:
        feat_info = json.load(f)
    FEATURE_COLS = feat_info["feature_cols"]
    SENSOR_COLS = feat_info["sensor_cols"]

    # Determine dropped sensors from what's NOT in sensor_cols
    all_sensors = [f"sensor_{i}" for i in range(1, 22)]
    DROPPED_SENSORS = [s for s in all_sensors if s not in SENSOR_COLS]
    print(f"Features: {len(FEATURE_COLS)}, Active sensors: {len(SENSOR_COLS)}")
    print(f"Dropped sensors: {DROPPED_SENSORS}")


# ---------------------------------------------------------------------------
# Feature Pipeline (mirrors Stage 2 logic)
# ---------------------------------------------------------------------------
def build_features(readings: list[dict]) -> pd.DataFrame:
    """
    Convert raw sensor readings into the feature vector expected by the model.
    
    This replicates the Stage 2 feature engineering pipeline:
      1. Create a DataFrame from the readings
      2. Drop low-variance sensors
      3. Compute rolling features (mean, std, slope)
      4. Apply the fitted scaler
    
    Returns a single-row DataFrame with the features for the last cycle.
    """
    # Build DataFrame
    df = pd.DataFrame(readings)

    # Add synthetic identifiers (single engine)
    df["unit_number"] = 1
    df["time_cycles"] = range(1, len(df) + 1)

    # Drop low-variance sensors
    for sensor in DROPPED_SENSORS:
        if sensor in df.columns:
            df = df.drop(columns=[sensor])

    # Compute rolling features for active sensors
    settings_cols = [c for c in df.columns if c.startswith("setting_")]
    active_sensors = [c for c in df.columns if c.startswith("sensor_")]

    new_cols = {}
    for sensor in active_sensors:
        for w in WINDOW_SIZES:
            prefix = f"{sensor}_w{w}"
            series = df[sensor]

            # Rolling mean
            new_cols[f"{prefix}_mean"] = series.rolling(window=w, min_periods=1).mean()

            # Rolling std
            new_cols[f"{prefix}_std"] = series.rolling(window=w, min_periods=1).std().fillna(0)

            # Rolling slope
            slopes = []
            for i in range(len(series)):
                start = max(0, i - w + 1)
                segment = series.iloc[start:i + 1].values
                if len(segment) < 2:
                    slopes.append(0.0)
                else:
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    slopes.append(coeffs[0])
            new_cols[f"{prefix}_slope"] = slopes

    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    # Select only the feature columns the model expects
    # (order matters for the scaler and model)
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing features after pipeline: {missing_cols}")

    feature_df = df[FEATURE_COLS].copy()

    # Scale using the fitted scaler
    feature_df[FEATURE_COLS] = SCALER.transform(feature_df[FEATURE_COLS])

    # Return the last row (most recent cycle) as prediction input
    return feature_df.iloc[[-1]]


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup."""
    load_model_artifacts()
    yield


app = FastAPI(
    title="CMAPSS RUL Prediction API",
    description="Predicts Remaining Useful Life (RUL) for turbofan engines "
                "using sensor data from the NASA CMAPSS dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the service status and whether the model is loaded.
    Useful for load balancers and deployment health monitoring.
    """
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION if MODEL is not None else "",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict Remaining Useful Life for an engine.
    
    Accepts a sequence of sensor readings (chronological order, oldest first)
    and returns the predicted number of cycles until failure.
    
    The prediction pipeline:
      1. Converts readings to a feature DataFrame
      2. Applies the same feature engineering as training (rolling stats)
      3. Scales features using the training scaler
      4. Runs the XGBoost model for prediction
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # Convert Pydantic models to dicts
        readings_dicts = [r.model_dump() for r in request.readings]

        # Build features
        features = build_features(readings_dicts)

        # Predict
        prediction = MODEL.predict(features.values)[0]

        # Clamp prediction to non-negative
        predicted_rul = max(0.0, float(prediction))

        # Add confidence note based on input length
        n_cycles = len(request.readings)
        if n_cycles < 5:
            note = f"Low confidence: only {n_cycles} cycle(s) provided. Recommend 20+ for best accuracy."
        elif n_cycles < 20:
            note = f"Moderate confidence: {n_cycles} cycles provided. Some rolling features use partial windows."
        else:
            note = f"Good confidence: {n_cycles} cycles provided."

        return PredictionResponse(
            predicted_rul=round(predicted_rul, 2),
            model_version=MODEL_VERSION,
            confidence_note=note,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
