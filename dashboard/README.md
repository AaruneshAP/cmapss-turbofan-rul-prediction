# CMAPSS Turbofan RUL Interactive Dashboard

An interactive Streamlit application for exploring NASA's CMAPSS turbofan engine degradation telemetry, tracking Remaining Useful Life (RUL) in real time, and visually comparing model calibration between the production **LSTM** network and the baseline **XGBoost** regressor.

**Live Streamlit App:** [https://cmapss-turbofan-rul-prediction-ijmejvqzgpdoyibtrxqhqk.streamlit.app/](https://cmapss-turbofan-rul-prediction-ijmejvqzgpdoyibtrxqhqk.streamlit.app/)  
**Live Serving API (Render):** [https://cmapss-rul-api-gfxh.onrender.com](https://cmapss-rul-api-gfxh.onrender.com)  
**Main Project Documentation:** [Full Project README](../README.md)

---

## Features

1. **Fleet Engine Selector**: Pick any of the 100 test turbofans from the CMAPSS FD001 dataset.
2. **Cycle Scrubber & Autoplay**: Scrub smoothly through operating cycles or hit **Play** to watch sensor wear and RUL depletion in real-time.
3. **Live API Integration**: Calls `POST /predict` on the deployed Render FastAPI layer with chronological 30-cycle sensor sequences, cached via `st.cache_data` for responsive performance.
4. **Visual Calibration & Overshoot Analysis**: Direct overlay of Ground Truth RUL, LSTM, and XGBoost baseline. Demonstrates that while RMSE is identical (~13.8), LSTM avoids dangerous late-stage overconfident predictions that cause high CMAPSS asymmetric penalties.
5. **Degradation Telemetry Trajectories**: Real-time micro-sparklines and delta stats for the 6 most degradation-sensitive sensors (`sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_11`, `sensor_12`).
6. **Model Benchmark Panel**: Dynamic metrics and interactive charts pulled directly from `experiments.json`.
7. **Graceful Free-Tier Cold-Start Handling**: Built-in health check and status feedback while Render containers spin up from sleep.

---

## Running Locally

From the repository root:

```bash
# Install lightweight dashboard requirements (no PyTorch/CUDA needed)
pip install -r dashboard/requirements.txt

# Run Streamlit dashboard
streamlit run dashboard/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Deploying to Streamlit Community Cloud

This folder is self-contained and ready for free deployment on [share.streamlit.io](https://share.streamlit.io):

1. Push this repository to GitHub:
   ```bash
   git push origin main
   ```
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **Create app** (or **New app**).
4. Select your repository: `AaruneshAP/Project1_cat`.
5. Set **Main file path** to: `dashboard/app.py`.
6. Click **Deploy!**

The deployment installs dependencies in under 15 seconds because heavy ML training libraries (`torch`, `xgboost`) are decoupled from the visualization frontend.

---

## Architecture & Data Flow

```
[User Browser]
       │
       ▼
[Streamlit Cloud Dashboard] (dashboard/app.py)
       │  • Loads clean test data (dashboard/data/test_fd001.parquet)
       │  • Loads Stage 3 XGBoost baseline (dashboard/data/xgb_predictions.json)
       │  • Extracts last 30 cycles of 21 sensors + 3 settings
       │
       ▼ (HTTPS POST /predict)
[Render Free-Tier FastAPI Container] (https://cmapss-rul-api-gfxh.onrender.com)
       │  • Scales telemetry with fitted StandardScaler
       │  • Runs PyTorch 2-layer LSTM forward pass
       │  • Returns predicted RUL + confidence note
       │
       ▼ (Cached in Streamlit @st.cache_data)
[Interactive Plotly Dual-Line Calibration Visualizer]
```

For the full end-to-end predictive maintenance project documentation (data exploration, feature engineering, modeling, Docker, and Render deployment), see the [Main Project README](../README.md).
