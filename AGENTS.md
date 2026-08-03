# CMAPSS Predictive Maintenance — Agent Build Instructions

## Objective
Build an end-to-end predictive maintenance system on NASA's CMAPSS turbofan
degradation dataset: data exploration → feature engineering → RUL (Remaining
Useful Life) regression model → FastAPI serving layer → Docker container →
free-tier deployment. Everything must run on free infrastructure (Kaggle/Colab
for training, Render free tier for deployment).

## Ground Rules for Agents
- Every stage must produce a runnable artifact (notebook, `.py` file, `Dockerfile`,
  etc.) — not just prose.
- Prefer plain, well-commented code over cleverness. This project will be
  explained by a human in an interview, so code should read like a tutorial.
- Log all metrics (RMSE, MAE) to a simple `experiments.json` or CSV so results
  are reproducible and comparable.
- Do not silently drop data or sensors — if a column is dropped, print/log why.
- Keep the whole pipeline dataset-agnostic where reasonable (FD001 first, but
  don't hardcode assumptions that break on FD002–FD004).

---

## Stage 1 — Data Acquisition & Exploration
**Environment:** Kaggle Notebook (free, no local setup)

1. Load `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` from the Kaggle
   "NASA CMAPSS Jet Engine Simulated Data" dataset.
2. Assign column names: `unit_number`, `time_cycles`, `setting_1..3`,
   `sensor_1..21`.
3. Checks to run and record:
   - Null value count (expect 0, CMAPSS is clean — confirm, don't assume).
   - Per-sensor `std()` — flag near-zero-variance sensors (typically sensors
     1, 5, 6, 10, 16, 18, 19 are flat/uninformative in FD001; verify rather
     than hardcode).
   - Per-unit max cycle count distribution (engine lifespans vary).
4. Construct training RUL: `RUL = max_cycle_for_unit - time_cycles`.
5. Produce plots: sensor trajectories over cycles for ~5 sample units,
   histogram of unit lifespans, correlation heatmap of sensors.
6. Save cleaned/labeled train and test frames as `train_fd001.parquet` and
   `test_fd001.parquet`.

**Output:** `01_exploration.ipynb`, `train_fd001.parquet`, `test_fd001.parquet`,
`eda_notes.md` (written findings: which sensors are dropped and why, general
degradation shape observed).

---

## Stage 2 — Feature Engineering
**Environment:** Kaggle/Colab

1. Drop zero/near-zero variance sensors identified in Stage 1.
2. Per unit, per remaining sensor, compute rolling-window features (window
   sizes 5 and 20 cycles):
   - Rolling mean
   - Rolling std
   - Rolling slope (linear regression coefficient over the window) as a
     degradation-trend proxy
3. Compute sensor-to-sensor correlation matrix; note any near-duplicate
   sensors (e.g., sensor_9 and sensor_14 are often highly correlated in
   CMAPSS) as candidates for redundancy removal — record which, don't
   auto-drop without logging the correlation value.
4. Clip training RUL at a ceiling (commonly 125 cycles) — this is standard
   practice because early-life RUL is not meaningfully predictable from
   sensor data; document this choice in comments.
5. Normalize/scale features (StandardScaler or MinMax), fit on train only,
   save the scaler object (`scaler.pkl`) for reuse at inference time.

**Output:** `02_feature_engineering.ipynb`, `features_train.parquet`,
`features_test.parquet`, `scaler.pkl`, `feature_notes.md` (physical
justification for kept/dropped features).

---

## Stage 3 — Modeling
**Environment:** Kaggle/Colab (enable free GPU only if training LSTM)

1. **Baseline — XGBoost regressor** predicting clipped RUL from engineered
   features.
   - Train/validation split by `unit_number` (never split by row — that leaks
     future cycles of the same engine into validation).
   - Tune `max_depth`, `n_estimators`, `learning_rate` via simple grid or
     `RandomizedSearchCV`.
   - Metrics: RMSE and MAE on held-out units, plus the CMAPSS-standard
     asymmetric scoring function (penalizes late predictions more than early
     ones) — implement this explicitly since it's a common interview point.
2. **Comparison — LSTM** on raw (non-aggregated) sequences using a sliding
   window per engine (e.g., last 30 cycles as one sample).
   - Same train/val split by unit.
   - Same metrics for direct comparison.
3. Log every run (model, params, RMSE, MAE, CMAPSS score) to
   `experiments.json`.
4. Select final model based on the CMAPSS score, not just RMSE, and justify
   the choice in `model_notes.md`.
5. Export the winning model:
   - XGBoost → `model.json` or `model.pkl`
   - LSTM → `model.pt` or `.h5`

**Output:** `03_modeling.ipynb`, `experiments.json`, `model_notes.md`,
final model artifact + `scaler.pkl` copied into `serving/model_artifacts/`.

---

## Stage 4 — API Wrapper
**Environment:** Local machine or Colab, exported to a standalone repo folder

1. Create `serving/app.py` — FastAPI app with:
   - `POST /predict` — accepts a JSON payload of the last N cycles of sensor
     readings for one engine, applies the saved scaler + feature pipeline,
     returns predicted RUL and the model version used.
   - `GET /health` — returns 200 + model load status.
2. Load the model and scaler once at startup (not per-request).
3. Input validation via Pydantic models — reject malformed sensor payloads
   with a clear 422 error rather than crashing.
4. Include `serving/requirements.txt` pinned to the exact library versions
   used in Stage 3 (mismatched xgboost/sklearn versions are the most common
   reason a saved model fails to load).

**Output:** `serving/app.py`, `serving/schemas.py`, `serving/requirements.txt`,
`serving/model_artifacts/` (model + scaler + feature list).

---

## Stage 5 — Containerization
**Environment:** Docker, local

1. Write `Dockerfile`:
   - Slim Python base image matching the training environment's Python
     version.
   - Copy `serving/` contents, install `requirements.txt`.
   - Expose port 8000, run via `uvicorn app:app --host 0.0.0.0 --port 8000`.
2. Build locally and verify:
   ```bash
   docker build -t cmapss-rul-api .
   docker run -p 8000:8000 cmapss-rul-api
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_payload.json
   ```
3. Include `sample_payload.json` (a realistic example request) in the repo
   for reviewers/interviewers to test with immediately.

**Output:** `Dockerfile`, `.dockerignore`, `sample_payload.json`, confirmed
local build+run log saved as `docker_verify.md`.

---

## Stage 6 — Deployment
**Environment:** Render free tier

1. Push the full repo (code + Dockerfile, NOT the raw CMAPSS data or large
   notebooks with output cells — use `.gitignore`) to GitHub.
2. Create a new Render Web Service from the repo, Docker runtime, free plan.
3. Confirm the deployed `/health` endpoint responds, then confirm `/predict`
   with the sample payload.
4. Document the cold-start behavior (free tier sleeps after inactivity;
   first request after sleep takes longer) in the README so it's not mistaken
   for a bug during a live demo.

**Output:** Live Render URL, `deploy_notes.md`.

---

## Stage 7 — README (Human-Written, Not Agent-Generated)
This stage is intentionally excluded from agent automation. The README is the
interview script — problem statement, why XGBoost vs. LSTM, what RUL means,
what the API does, what would improve with more time/data. Agents should
leave a `README_TEMPLATE.md` with section headers only (no filled content) so
the human writes the substance themselves.

**Output:** `README_TEMPLATE.md` (headers only — Problem, Data, Approach,
Why XGBoost, API Design, Deployment, Limitations & Future Work).

---

## Final Repo Structure
```
cmapss-predictive-maintenance/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── data/                      # gitignored, local/Kaggle only
├── serving/
│   ├── app.py
│   ├── schemas.py
│   ├── requirements.txt
│   └── model_artifacts/
├── Dockerfile
├── .dockerignore
├── sample_payload.json
├── experiments.json
├── eda_notes.md
├── feature_notes.md
├── model_notes.md
├── deploy_notes.md
└── README_TEMPLATE.md
```
