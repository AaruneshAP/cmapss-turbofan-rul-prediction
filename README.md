# CMAPSS Turbofan RUL Prediction

An end-to-end predictive maintenance system built on NASA's CMAPSS turbofan degradation dataset.

Given a stream of engine sensor readings, the system predicts **Remaining Useful Life (RUL)** — the number of cycles until the engine fails — and serves that prediction through a production-ready REST API deployed in Docker on Render.

**Try it live (interactive dashboard):** [Streamlit Cloud Dashboard](https://cmapss-turbofan-rul-prediction-ijmejvqzgpdoyibtrxqhqk.streamlit.app/)
**Live API docs (interactive):** https://cmapss-rul-api-gfxh.onrender.com/docs
**Health check:** https://cmapss-rul-api-gfxh.onrender.com/health

The interactive Streamlit dashboard lets a visitor pick any of the 100 test turbofan engines, scrub through its operational cycles (or autoplay with live telemetry), and watch real-time predictions from the live API evolve against ground-truth RUL. It visually validates the project's central finding: while XGBoost and LSTM achieve near-identical RMSE (~13.8), XGBoost exhibits dangerous overconfident predictions near end-of-life that the sequential LSTM model successfully avoids.

> The bare domain (`.../` with no path) intentionally returns a small JSON
> pointer to these documentation and health links rather than a 404 — see the root route in
> `serving/app.py`.

> Runs on Render's free tier, which sleeps after 15 minutes of inactivity —
> the first request after idle time can take 30–60s to respond while the
> container wakes up. Send a `/health` check a minute or two before a live
> demo to warm it up.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Repo Structure](#3-repo-structure)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Feature Engineering](#5-feature-engineering)
6. [Modelling & Results](#6-modelling--results)
7. [Why LSTM Over XGBoost](#7-why-lstm-over-xgboost)
8. [API Design](#8-api-design)
9. [Running Locally](#9-running-locally)
10. [Docker](#10-docker)
11. [Deployment (Render)](#11-deployment-render)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Problem Statement

Unplanned turbofan engine failure is expensive and dangerous. Predictive maintenance aims to replace "fix it when it breaks" with "fix it just before it breaks" by forecasting RUL from sensor telemetry.

**RUL** = (max cycle for this engine) − (current cycle)

The challenge: sensors read identically in early engine life regardless of total lifespan, so RUL is only learnable from sensor data in roughly the last 125 cycles of degradation. This is why the training labels are clipped at 125 cycles — more on this in [Feature Engineering](#5-feature-engineering).

---

## 2. Dataset

**NASA CMAPSS — Commercial Modular Aero-Propulsion System Simulation**
Source: [NASA PCoE direct mirror](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) — downloaded directly rather than via Kaggle's API to avoid authentication/token setup for a one-off download.

This project uses the **FD001** subset (single operating condition, one fault mode — HPC degradation).

| Split | Engines | Rows | Cycles per engine |
|-------|---------|------|-------------------|
| Train | 100 | 20,631 | 128 – 362 (mean 206) |
| Test | 100 | 13,096 | variable (RUL file provides ground truth) |

**Columns:** `unit_number`, `time_cycles`, `setting_1..3`, `sensor_1..21`

**Key facts confirmed during EDA (not assumed):**
- Zero null values — CMAPSS is fully synthetic and clean.
- 7 of 21 sensors are constant or near-constant across all engines (std < 0.01) and carry no predictive signal. These are dropped.
- Engine lifespan distribution is roughly bell-shaped with slight right skew (std = 46 cycles).

---

## 3. Repo Structure

```
cmapss-predictive-maintenance/
├── dashboard/                      # Interactive Streamlit application
│   ├── app.py                      # UI with trajectory charts, autoplay, and overlays
│   ├── data_utils.py               # Data loaders, caching, and live API client
│   ├── requirements.txt            # Lightweight dashboard dependencies
│   └── data/                       # Packaged test data & precomputed baseline
├── notebooks/
│   ├── 01_exploration.ipynb        # EDA, sensor analysis, RUL construction
│   ├── 02_feature_engineering.ipynb# Rolling features, scaling, parquet output
│   └── 03_modeling.py              # XGBoost + LSTM training, metric logging
├── serving/
│   ├── app.py                      # FastAPI application
│   ├── schemas.py                  # Pydantic request/response models
│   ├── requirements.txt            # Pinned deps (torch installed separately)
│   └── model_artifacts/
│       ├── lstm_model.pt           # Trained LSTM weights + metadata
│       ├── scaler.pkl              # Fitted StandardScaler (Stage 2)
│       ├── feature_list.json       # Authoritative column list from checkpoint
│       └── model_selection.json    # Experiment log for both models
├── data/                           # .gitignored — download locally or on Kaggle
├── Dockerfile
├── .dockerignore
├── sample_payload.json             # 30-cycle request ready for curl/Postman
├── experiments.json                # All training run metrics
├── eda_notes.md
├── feature_notes.md
├── model_notes.md
└── deploy_notes.md
```

---

## 4. Pipeline Overview

```
Raw CMAPSS txt files
        │
        ▼
Stage 1 — Exploration (01_exploration.ipynb)
  • Column naming, null checks, sensor variance analysis
  • RUL construction: RUL = max_cycle − time_cycles
  • Outputs: train_fd001.parquet, test_fd001.parquet
        │
        ▼
Stage 2 — Feature Engineering (02_feature_engineering.ipynb)
  • Drop 7 zero/near-zero variance sensors
  • Rolling mean, std, slope (windows 5 and 20) per retained sensor
  • Clip training RUL at 125 cycles
  • Fit StandardScaler on train only → scaler.pkl
  • Outputs: features_train.parquet, features_test.parquet
        │
        ▼
Stage 3 — Modelling (03_modeling.py)
  • XGBoost: flat features, unit-aware train/val split, RandomizedSearchCV
  • LSTM: raw sensor sequences (seq_length=30), 2-layer LSTM, early stopping
  • Log RMSE, MAE, CMAPSS asymmetric score → experiments.json
  • Winner: LSTM (lower CMAPSS score)
        │
        ▼
Stage 4 — FastAPI (serving/app.py)
  • POST /predict — sequence → scaled tensor → LSTM → RUL
  • GET  /health  — readiness check
        │
        ▼
Stage 5 — Docker (Dockerfile)
  • python:3.13-slim base
  • torch installed from CPU-only wheel index (avoids ~2 GB CUDA download)
  • Remaining deps from requirements.txt
        │
        ▼
Stage 6 — Render (free tier)
  • Docker runtime, $PORT env var, auto-sleep after 15 min inactivity
```

---

## 5. Feature Engineering

### Sensor Drops (7 removed)

| Sensor | Std Dev | Reason |
|--------|---------|--------|
| sensor_1 | 0.000000 | Constant — no information content |
| sensor_5 | ~5e-15 | Effectively constant (float noise only) |
| sensor_6 | 0.0014 | Negligible variation |
| sensor_10 | 0.000000 | Constant |
| sensor_16 | ~4e-18 | Effectively constant |
| sensor_18 | 0.000000 | Constant |
| sensor_19 | 0.000000 | Constant |

### Retained Features (17 total input columns)

`setting_1`, `setting_2`, `setting_3` + 14 active sensors:
`sensor_2` (fan inlet temp), `sensor_3` (HPC outlet temp), `sensor_4` (LPT outlet temp), `sensor_7` (HPC static pressure), `sensor_8` (fuel-flow ratio), `sensor_9` (corrected fan speed), `sensor_11` (HPT outlet temp), `sensor_12`, `sensor_13`, `sensor_14` (corrected core speed), `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`

> **Note:** sensor_9 and sensor_14 are highly correlated (r = 0.963 — corrected fan vs core speed). Both are retained because their slight decorrelation during degradation carries diagnostic signal, but a production system might drop one.

### Rolling Features (XGBoost only)

For each of the 14 active sensors, rolling windows of **5** and **20** cycles compute:
- **mean** — smoothed current state, reduces cycle noise
- **std** — signal instability; engines often show increasing sensor variance approaching failure
- **slope** — linear regression coefficient over the window; directly measures degradation rate

14 sensors × 2 windows × 3 statistics = 84 rolling features + 14 raw + 3 settings = **101 features total** (XGBoost path).

The LSTM sees **raw columns only** (17 features per timestep) — it learns temporal patterns itself from the 30-cycle input window.

### RUL Clipping at 125 Cycles

Training RUL is clipped at 125. In early engine life (say, cycle 10 of a 350-cycle engine), sensor readings are indistinguishable from cycle 10 of a 200-cycle engine — the degradation hasn't started yet. Clipping forces the model to learn from the portion of the trajectory where degradation is actually observable in the data. This is standard practice in CMAPSS benchmarks.

### Scaling

`StandardScaler` (zero mean, unit variance) fitted on **training data only**. The same fitted scaler object (`scaler.pkl`) is used at inference time to prevent data leakage.

---

## 6. Modelling & Results

### Train / Validation Split

Split is **by `unit_number`** (80 train / 20 validation units). Splitting by row would leak future cycles of the same engine into validation, making metrics look better than they are. `GroupKFold` with `unit_number` as the group key enforces this during cross-validation.

### Metrics

**RMSE / MAE** are standard regression metrics.

**CMAPSS Asymmetric Score** is the domain-specific metric defined in the NASA paper:

$$s = \sum_{i} \begin{cases} e^{-d_i/13} - 1 & \text{if } d_i < 0 \quad \text{(late / conservative)} \\ e^{d_i/10} - 1 & \text{if } d_i \geq 0 \quad \text{(early / optimistic)} \end{cases}$$

where $d_i = \hat{y}_i - y_i$ (predicted minus actual RUL). The **asymmetry** reflects real-world cost: predicting too much remaining life (engine fails unexpectedly) is far more dangerous than predicting too little (unnecessary early maintenance). **Lower score = better.**

### Results

| Model | Val RMSE | Val MAE | Val CMAPSS Score | Train Time |
|-------|----------|---------|-----------------|------------|
| XGBoost | 13.82 | 9.93 | 17,865 | 888 s |
| **LSTM** | **13.86** | **10.38** | **12,253** ✓ | **7 s** |

**Selected model: LSTM** — wins decisively on CMAPSS score (31% lower), which is the metric that matters for the safety-critical use case.

### XGBoost Hyperparameters (best found by `RandomizedSearchCV`)

```json
{
  "n_estimators": 500, "max_depth": 5, "learning_rate": 0.01,
  "subsample": 0.7, "colsample_bytree": 0.7,
  "reg_alpha": 1.0, "reg_lambda": 5.0, "min_child_weight": 5
}
```

### LSTM Architecture

```
Input: (batch, 30, 17)  ← 30 cycles × 17 features
  └─ nn.LSTM(17 → 64, num_layers=2, dropout=0.3, batch_first=True)
  └─ Last timestep hidden state: (batch, 64)
  └─ FC: Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1)
Output: (batch,)  ← scalar RUL per engine
```

Training: Adam (lr=0.001), MSE loss, batch size 256, up to 50 epochs with patience=10 early stopping. Converged at epoch 20.

---

## 7. Why LSTM Over XGBoost

The CMAPSS score is the deciding metric. Here's the reasoning:

**LSTM wins because it captures temporal order natively.** XGBoost treats each row as independent; rolling features are an approximation of temporal context. The LSTM processes the actual sequence of 30 cycles in order, allowing it to weight how readings are *changing* — not just what they are. That temporal sensitivity produces predictions that are better calibrated against late failures (lower asymmetric penalty).

**XGBoost's strengths** (relevant for interview discussion):
- Interpretable via feature importances
- Tiny model file (~565 KB vs ~226 KB for LSTM, but LSTM needs PyTorch ~200 MB)
- Retraining takes seconds; LSTM needs GPU access for practical retraining
- Competitive RMSE (13.82 vs 13.86) — the difference in raw error is negligible

In a safety-critical production system, the CMAPSS score's asymmetric penalty is the right objective, so LSTM is the correct choice here.

---

## 8. API Design

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service liveness — returns model load status |
| `POST` | `/predict` | Returns predicted RUL for one engine |

### POST /predict

**Request body:**
```json
{
  "readings": [
    {
      "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0,
      "sensor_1": 518.67, "sensor_2": 641.82, "..": "...",
      "sensor_21": 23.419
    }
    // ... up to 500 readings, chronological order oldest→newest
  ]
}
```

All 21 sensors and 3 settings are accepted (matching the raw CMAPSS schema). Dropped/uninformative sensors are included in the payload for simplicity — the serving layer extracts only the 17 columns the LSTM was trained on.

**Serving pipeline:**
1. Extract the 17 active columns in training order (from checkpoint metadata — not hardcoded)
2. Apply the fitted `StandardScaler` column-by-column
3. Take the last 30 cycles; zero-pad the front if fewer than 30 are supplied
4. Run `torch.no_grad()` forward pass → scalar output
5. Clamp to `[0, ∞)` and return

**Response:**
```json
{
  "predicted_rul": 112.45,
  "model_version": "lstm-fd001-v1",
  "confidence_note": "Good confidence: 30 cycle(s) provided (≥ seq_length=30)."
}
```

**Error handling:**
- Missing sensor columns → `422 Unprocessable Entity` with field-level detail (Pydantic)
- Payload > 500 readings → `422` (validator)
- Model not loaded → `503 Service Unavailable`
- Runtime failure → `500` with message

### Interactive Docs

FastAPI auto-generates Swagger UI at `/docs` and ReDoc at `/redoc` — useful for exploring the API without writing curl commands.

---

## 9. Running Locally

**Prerequisites:** Python 3.13, PyTorch (CPU), the packages in `serving/requirements.txt`.

```bash
# Install deps
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r serving/requirements.txt

# Start the server
cd serving
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Health check
curl http://localhost:8000/health

# Prediction (uses the 30-cycle sample payload)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

The `--reload` flag restarts the server on code changes — useful during development.

---

## 10. Docker

```bash
# Build (torch is installed from CPU-only wheel — no CUDA download)
docker build -t cmapss-rul-api .

# Run (defaults to port 8000)
docker run -p 8000:8000 cmapss-rul-api

# Run on a custom port (mirrors Render's $PORT behaviour)
docker run -e PORT=9000 -p 9000:9000 cmapss-rul-api

# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

**Why torch is installed separately from `requirements.txt`:**

PyPI's default torch index serves CUDA wheels (~2 GB). The Dockerfile uses `--index-url https://download.pytorch.org/whl/cpu` to download the CPU-only wheel (~200 MB instead). This is done in a separate `RUN` layer before the `requirements.txt` install so Docker caches the large torch download independently of application code changes.

---

## 11. Deployment (Render)

The live API is deployed on Render's free tier as a Docker web service.

**Live URL:** `https://cmapss-rul-api-gfxh.onrender.com`

```bash
# Wake the service before a demo (cold-start takes 30–60 s)
curl https://cmapss-rul-api-gfxh.onrender.com/health

# Prediction
curl -X POST https://cmapss-rul-api-gfxh.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

### ⚠️ Cold-Start Behaviour

Render's free tier **spins down the container after 15 minutes of inactivity**. The first request after sleep takes **30–60 seconds** (container restart + Python boot + model load). This is **expected behaviour**, not a bug.

**Before a live demo:** Send a health check request 1–2 minutes in advance to warm the service. Mentioning this in an interview demonstrates awareness of free-tier cloud constraints.

### Deploying Your Own Instance

1. Push to GitHub (the `data/` directory is gitignored — only code and artifacts are pushed)
2. Create a new Render Web Service → Docker runtime → connect your repo → free plan
3. Render injects `$PORT` automatically; the `CMD` reads it via `${PORT:-8000}`

See [`deploy_notes.md`](deploy_notes.md) for full step-by-step instructions.

---

## 12. Limitations & Future Work

### Current Limitations

- **FD001 only** — single operating condition, single fault mode (HPC degradation). FD002–FD004 introduce multiple operating conditions and fault modes; the current feature pipeline would need adaptation.
- **No online learning** — the model is static. Real engines accumulate new failure data continuously; a production system should support periodic retraining.
- **Free tier cold starts** — not suitable for latency-sensitive applications without upgrading to a paid host.
- **sensor_9 / sensor_14 redundancy** (r = 0.963) — both are retained; dropping one might reduce overfitting slightly.

### What Would Improve With More Time / Data

| Improvement | Expected Benefit |
|-------------|-----------------|
| Train on FD002–FD004 | Generalisation across operating conditions |
| Ensemble XGBoost + LSTM | Better calibration than either alone |
| Attention mechanism on LSTM | Interpretability — which timesteps matter most? |
| Bayesian uncertainty estimates | Confidence intervals on RUL predictions |
| Streaming `/predict` via WebSocket | Real-time monitoring rather than batch requests |
| Model retraining pipeline (Airflow / Prefect) | Keep model current as new failure data arrives |

---

## Acknowledgements

- NASA Ames Research Center for the CMAPSS dataset
- [Saxena et al. (2008)](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) — original dataset paper and asymmetric scoring function definition
