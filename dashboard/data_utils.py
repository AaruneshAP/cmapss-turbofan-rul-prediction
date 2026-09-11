"""
Data utilities and API client for the CMAPSS RUL Streamlit Dashboard.
"""

import os
import json
import time
import requests
import pandas as pd
import streamlit as st

LIVE_API_DEFAULT_URL = "https://cmapss-rul-api-gfxh.onrender.com"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(CURRENT_DIR, "data", "test_fd001.parquet")
XGB_PREDS_PATH = os.path.join(CURRENT_DIR, "data", "xgb_predictions.json")
EXPERIMENTS_PATH = os.path.join(PROJECT_ROOT, "experiments.json")

# Physical sensor descriptions for the top informative sensors identified in EDA
SENSOR_METADATA = {
    "sensor_2": {
        "label": "T2 — Total Temp at Fan Inlet",
        "unit": "°R",
        "description": "Inlet air temperature; increases gradually as fan and compressor aerodynamic efficiency degrades.",
        "color": "#38BDF8",  # Sky Blue
    },
    "sensor_3": {
        "label": "T3 — Total Temp at HPC Outlet",
        "unit": "°R",
        "description": "High-Pressure Compressor discharge temperature; primary thermal indicator of compressor wear.",
        "color": "#F43F5E",  # Rose
    },
    "sensor_4": {
        "label": "T4 — Total Temp at LPT Outlet",
        "unit": "°R",
        "description": "Low-Pressure Turbine discharge temperature; captures downstream thermal loading from degraded core.",
        "color": "#FB923C",  # Orange
    },
    "sensor_7": {
        "label": "Ps30 — HPC Outlet Static Pressure",
        "unit": "psia",
        "description": "High-Pressure Compressor static pressure; drops steadily as blade clearances and seals wear.",
        "color": "#A855F7",  # Purple
    },
    "sensor_11": {
        "label": "T50 — Total Temp at HPT Outlet",
        "unit": "°R",
        "description": "High-Pressure Turbine temperature; shows marked upward acceleration in the final 50 cycles.",
        "color": "#EC4899",  # Pink
    },
    "sensor_12": {
        "label": "Nf — Fan Rotational Speed",
        "unit": "rpm",
        "description": "Physical fan rotational speed; decreases as compressor fouling increases aerodynamic resistance.",
        "color": "#34D399",  # Emerald
    },
}

INFORMATIVE_SENSORS = list(SENSOR_METADATA.keys())


@st.cache_data(show_spinner=False)
def load_test_data() -> pd.DataFrame:
    """Load clean test dataset containing 100 test engines with ground truth RUL."""
    if os.path.exists(DATA_PATH):
        return pd.read_parquet(DATA_PATH)
    # Fallback to project root data if available
    root_data = os.path.join(PROJECT_ROOT, "data", "test_fd001.parquet")
    if os.path.exists(root_data):
        return pd.read_parquet(root_data)
    raise FileNotFoundError(f"Test dataset not found at {DATA_PATH} or {root_data}")


@st.cache_data(show_spinner=False)
def load_xgb_predictions() -> dict:
    """Load precomputed XGBoost predictions for all test engines."""
    if os.path.exists(XGB_PREDS_PATH):
        with open(XGB_PREDS_PATH, "r") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_experiments() -> list[dict]:
    """Load Stage 3 model training and benchmark experiments."""
    if os.path.exists(EXPERIMENTS_PATH):
        with open(EXPERIMENTS_PATH, "r") as f:
            return json.load(f)
    # Fallback default from Stage 3 if file not found
    return [
        {
            "model": "XGBoost",
            "val_rmse": 13.8229,
            "val_mae": 9.9275,
            "val_cmapss_score": 17864.82,
            "training_time_s": 887.8,
        },
        {
            "model": "LSTM",
            "val_rmse": 13.8624,
            "val_mae": 10.3839,
            "val_cmapss_score": 12252.56,
            "training_time_s": 7.21,
        },
    ]


def check_api_health(api_url: str = LIVE_API_DEFAULT_URL, timeout: float = 3.0) -> dict:
    """
    Check the live FastAPI serving layer health.
    Returns status dict with connectivity and model details.
    """
    endpoint = f"{api_url.rstrip('/')}/health"
    try:
        resp = requests.get(endpoint, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "online": True,
                "status": data.get("status", "healthy"),
                "model_loaded": data.get("model_loaded", False),
                "model_version": data.get("model_version", "lstm-fd001-v1"),
                "error": None,
            }
        return {
            "online": False,
            "status": f"HTTP {resp.status_code}",
            "model_loaded": False,
            "model_version": "",
            "error": f"API returned status {resp.status_code}",
        }
    except requests.exceptions.Timeout:
        return {
            "online": False,
            "status": "sleeping",
            "model_loaded": False,
            "model_version": "",
            "error": "Connection timed out. Free-tier container is likely asleep.",
        }
    except Exception as e:
        return {
            "online": False,
            "status": "unreachable",
            "model_loaded": False,
            "model_version": "",
            "error": str(e),
        }


def build_sensor_payload(engine_df: pd.DataFrame, current_cycle: int, max_cycles: int = 30) -> list[dict]:
    """
    Extract the last up to `max_cycles` telemetry readings up to `current_cycle`.
    Matches the schema expected by POST /predict on the live FastAPI service.
    """
    history = engine_df[engine_df["time_cycles"] <= current_cycle].tail(max_cycles)
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    setting_cols = ["setting_1", "setting_2", "setting_3"]
    cols_to_extract = setting_cols + sensor_cols

    # Ensure all required columns are present (fill 0.0 if missing)
    for c in cols_to_extract:
        if c not in history.columns:
            history[c] = 0.0

    readings = history[cols_to_extract].to_dict(orient="records")
    return readings


@st.cache_data(show_spinner=False, max_entries=2000)
def query_live_api(api_url: str, unit_number: int, current_cycle: int, payload_json_str: str) -> dict:
    """
    Query the live FastAPI endpoint for predicted RUL.
    Cached by (api_url, unit_number, current_cycle) to minimize Render free tier traffic.
    """
    endpoint = f"{api_url.rstrip('/')}/predict"
    payload = json.loads(payload_json_str)

    try:
        resp = requests.post(endpoint, json=payload, timeout=25.0)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "predicted_rul": float(data.get("predicted_rul", 0.0)),
                "model_version": data.get("model_version", "lstm-fd001-v1"),
                "confidence_note": data.get("confidence_note", ""),
                "error": None,
            }
        return {
            "success": False,
            "predicted_rul": None,
            "model_version": "",
            "confidence_note": "",
            "error": f"API error {resp.status_code}: {resp.text}",
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "predicted_rul": None,
            "model_version": "",
            "confidence_note": "",
            "error": "Request timed out while waiting for Render free-tier container.",
        }
    except Exception as e:
        return {
            "success": False,
            "predicted_rul": None,
            "model_version": "",
            "confidence_note": "",
            "error": str(e),
        }
