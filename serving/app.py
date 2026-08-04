"""
Stage 4 — FastAPI Serving Layer
=================================
Serves CMAPSS RUL predictions via a REST API.

Endpoints:
  POST /predict — accepts sensor readings, returns predicted RUL
  GET  /health  — returns service health and model load status

Model: LSTM (selected over XGBoost based on CMAPSS asymmetric score).
  - Input shape : (1, seq_length, n_sensor_features)
  - seq_length  : read from lstm_model.pt checkpoint (default 30 cycles)
  - Features    : raw sensor + settings columns only (no rolling stats)
  - Scaler      : StandardScaler fitted in Stage 2, applied per-feature

The model, scaler, and sequence metadata are loaded once at startup.
"""

import os
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from schemas import PredictionRequest, PredictionResponse, HealthResponse

# ---------------------------------------------------------------------------
# Globals — populated at startup, never mutated afterwards
# ---------------------------------------------------------------------------
MODEL: nn.Module | None = None
SCALER = None
SENSOR_COLS: list[str] = []   # raw sensor + settings cols the LSTM was trained on
SEQ_LENGTH: int = 30          # sliding-window length used in training
MODEL_VERSION: str = "lstm-fd001-v1"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")


# ---------------------------------------------------------------------------
# LSTM architecture — must mirror the definition in 03_modeling.py exactly
# so that load_state_dict() succeeds.
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """
    2-layer LSTM with a small fully-connected head.

    input_size  : number of features per time step
    hidden_size : LSTM hidden units (64, same as training)
    num_layers  : stacked LSTM layers (2, same as training)
    dropout     : dropout between LSTM layers (0.3, same as training)
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Take only the last time step's hidden state
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze(-1)


# ---------------------------------------------------------------------------
# Startup: load model artifacts
# ---------------------------------------------------------------------------
def load_model_artifacts() -> None:
    """
    Load the LSTM model, scaler, and sequence metadata at startup.

    Reading seq_length and sensor_cols from the checkpoint itself keeps the
    serving layer in sync with whatever was saved at the end of training —
    no need to hard-code values here.
    """
    global MODEL, SCALER, SENSOR_COLS, SEQ_LENGTH

    # --- Load LSTM checkpoint ---
    model_path = os.path.join(ARTIFACTS_DIR, "lstm_model.pt")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Metadata stored inside the checkpoint by 03_modeling.py
    input_size: int  = checkpoint["input_size"]   # number of features per step
    SENSOR_COLS      = checkpoint["sensor_cols"]  # ordered list of feature names
    SEQ_LENGTH       = checkpoint.get("seq_length", 30)  # fallback to 30 if absent

    # Reconstruct the model and load weights
    MODEL = LSTMModel(input_size=input_size)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.eval()  # inference mode — disables dropout

    print(f"Loaded LSTM model from {model_path}")
    print(f"  input_size : {input_size}")
    print(f"  seq_length : {SEQ_LENGTH}")
    print(f"  sensor_cols: {SENSOR_COLS}")

    # --- Load scaler (fitted on train data in Stage 2) ---
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    SCALER = joblib.load(scaler_path)
    print(f"Loaded scaler from {scaler_path}")

    # --- Load feature_list.json for cross-validation only (optional) ---
    feat_path = os.path.join(ARTIFACTS_DIR, "feature_list.json")
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            feat_info = json.load(f)
        # Warn if checkpoint sensor_cols diverges from feature_list.json
        fl_sensor_cols = feat_info.get("sensor_cols", [])
        if set(SENSOR_COLS) != set(fl_sensor_cols):
            print(
                "[WARN] sensor_cols in checkpoint differ from feature_list.json. "
                "Using checkpoint values (they are authoritative)."
            )


# ---------------------------------------------------------------------------
# Preprocessing: raw readings → (1, seq_length, n_features) tensor
# ---------------------------------------------------------------------------
def build_lstm_sequence(readings: list[dict]) -> torch.Tensor:
    """
    Convert a list of raw sensor readings into the LSTM input tensor.

    The LSTM was trained on sliding windows of shape (seq_length, n_features)
    over *raw* sensor + settings columns (no rolling aggregates — the LSTM
    learns temporal patterns itself).

    Steps:
      1. Extract only the columns in SENSOR_COLS (ordered).
      2. Apply the fitted StandardScaler (same object used in Stage 2).
      3. Take the last SEQ_LENGTH rows; zero-pad at the front if fewer rows
         are available (identical to how training sequences were built).
      4. Return shape (1, SEQ_LENGTH, n_features) as a float32 tensor.

    Zero-padding at the front means the model will see neutral (zero) values
    for the "missing" early cycles, which is a safe default and avoids
    extrapolation errors.
    """
    import pandas as pd

    # Build a DataFrame from raw readings
    df = pd.DataFrame(readings)

    # Validate that every required column is present
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Payload is missing required sensor/setting columns: {missing}. "
            f"Expected: {SENSOR_COLS}"
        )

    # Extract features in the exact order the model was trained on
    raw = df[SENSOR_COLS].values.astype(np.float32)  # shape: (n_cycles, n_features)

    # Scale using the Stage-2 scaler.
    # The scaler was fitted on all feature_cols (including rolling ones), so we
    # use transform() on just the sensor columns — sklearn scalers operate
    # column-by-column and the indices align because SENSOR_COLS is a
    # contiguous prefix of feature_cols.
    # To be safe, we fit only the relevant columns from the scaler via the
    # feature names the scaler knows about.
    try:
        # sklearn >= 1.0: use feature_names_in_ if available
        if hasattr(SCALER, "feature_names_in_"):
            scaler_cols = list(SCALER.feature_names_in_)
            col_indices = [scaler_cols.index(c) for c in SENSOR_COLS]
            mean_ = SCALER.mean_[col_indices]
            scale_ = SCALER.scale_[col_indices]
            raw = (raw - mean_) / scale_
        else:
            # Fallback: assume SENSOR_COLS align with the first N scaler columns
            n = len(SENSOR_COLS)
            raw = (raw - SCALER.mean_[:n]) / SCALER.scale_[:n]
    except Exception as e:
        raise ValueError(f"Scaling failed: {e}") from e

    n_cycles, n_features = raw.shape
    seq = np.zeros((SEQ_LENGTH, n_features), dtype=np.float32)

    if n_cycles >= SEQ_LENGTH:
        # Use the most recent SEQ_LENGTH cycles
        seq[:] = raw[-SEQ_LENGTH:]
    else:
        # Zero-pad the front; fill the tail with whatever we have
        seq[-n_cycles:] = raw

    # Add batch dimension → (1, SEQ_LENGTH, n_features)
    tensor = torch.from_numpy(seq).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup."""
    load_model_artifacts()
    yield


app = FastAPI(
    title="CMAPSS RUL Prediction API",
    description=(
        "Predicts Remaining Useful Life (RUL) for turbofan engines "
        "using sensor data from the NASA CMAPSS dataset. "
        "Model: LSTM (selected over XGBoost on CMAPSS asymmetric score)."
    ),
    version="2.0.0",
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
    Predict Remaining Useful Life for an engine using the LSTM model.

    Accepts a chronological sequence of sensor readings (oldest first) and
    returns the predicted number of cycles until failure.

    The prediction pipeline:
      1. Extracts raw sensor + settings columns (no rolling features needed).
      2. Scales each feature column using the Stage-2 StandardScaler.
      3. Builds a (1, seq_length, n_features) tensor — zero-padded at the
         front if fewer than seq_length cycles are provided.
      4. Runs a single forward pass through the LSTM (no gradient computation).
      5. Clamps the output to [0, ∞) and returns it.

    Confidence note:
      - < seq_length cycles → low confidence (sequence is partly zero-padded).
      - ≥ seq_length cycles → good confidence (full window available).
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # Convert Pydantic models to plain dicts
        readings_dicts = [r.model_dump() for r in request.readings]

        # Build the LSTM input tensor
        sequence_tensor = build_lstm_sequence(readings_dicts)

        # Run inference (no gradient tracking needed at serving time)
        with torch.no_grad():
            raw_pred = MODEL(sequence_tensor)  # shape: (1,)

        predicted_rul = max(0.0, float(raw_pred.item()))

        # Confidence note based on available history vs. training seq_length
        n_cycles = len(request.readings)
        if n_cycles < SEQ_LENGTH:
            note = (
                f"Low confidence: only {n_cycles} cycle(s) provided, but the "
                f"LSTM was trained on windows of {SEQ_LENGTH} cycles. "
                f"The missing early cycles are zero-padded. "
                f"Recommend providing at least {SEQ_LENGTH} cycles."
            )
        else:
            note = (
                f"Good confidence: {n_cycles} cycle(s) provided "
                f"(≥ seq_length={SEQ_LENGTH})."
            )

        return PredictionResponse(
            predicted_rul=round(predicted_rul, 2),
            model_version=MODEL_VERSION,
            confidence_note=note,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
