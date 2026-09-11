"""
CMAPSS Turbofan RUL Interactive Dashboard
==========================================
Visualizes engine degradation, compares real-time live API predictions against
ground truth RUL, and contrasts LSTM calibration with XGBoost baseline.
"""

import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_utils import (
    LIVE_API_DEFAULT_URL,
    SENSOR_METADATA,
    INFORMATIVE_SENSORS,
    load_test_data,
    load_xgb_predictions,
    load_experiments,
    check_api_health,
    build_sensor_payload,
    query_live_api,
)

# ---------------------------------------------------------------------------
# Page Configuration & CSS Design Tokens
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CMAPSS Turbofan RUL Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern Glassmorphism & Dark Palette */
:root {
    --bg-card: rgba(30, 41, 59, 0.7);
    --border-card: rgba(148, 163, 184, 0.15);
    --accent-cyan: #38BDF8;
    --accent-emerald: #10B981;
    --accent-amber: #F59E0B;
    --accent-rose: #F43F5E;
}

.stApp {
    background-color: #0B0F19;
    color: #F1F5F9;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid var(--border-card);
}

/* Metric Cards */
.kpi-container {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(8px);
}
.kpi-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 4px;
}

/* Status Badges */
.badge-online {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.badge-sleeping {
    background-color: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.badge-error {
    background-color: rgba(244, 63, 94, 0.15);
    color: #F43F5E;
    border: 1px solid rgba(244, 63, 94, 0.3);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Callout Box */
.callout-box {
    background: rgba(15, 23, 42, 0.8);
    border-left: 4px solid var(--accent-cyan);
    border-radius: 6px;
    padding: 12px 16px;
    margin: 16px 0;
    font-size: 0.9rem;
    color: #CBD5E1;
}

/* Footer Styling */
.footer-container {
    border-top: 1px solid var(--border-card);
    padding: 24px 0 12px 0;
    margin-top: 36px;
    font-size: 0.85rem;
    color: #64748B;
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False
if "current_cycle" not in st.session_state:
    st.session_state.current_cycle = 30
if "selected_unit" not in st.session_state:
    st.session_state.selected_unit = 34  # Default to Unit 34 (rich degradation profile ending near failure)
if "api_url" not in st.session_state:
    st.session_state.api_url = LIVE_API_DEFAULT_URL
if "api_status_cache" not in st.session_state:
    st.session_state.api_status_cache = None
if "last_health_check" not in st.session_state:
    st.session_state.last_health_check = 0

# ---------------------------------------------------------------------------
# Load Cached Data
# ---------------------------------------------------------------------------
try:
    test_df = load_test_data()
    xgb_preds_all = load_xgb_predictions()
    experiments_data = load_experiments()
except Exception as e:
    st.error(f"Failed loading local test artifacts: {e}")
    st.stop()

available_units = sorted(test_df["unit_number"].unique())

# ---------------------------------------------------------------------------
# Sidebar: Controls & Live API Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Telemetry Controls")

    # 1. Engine Selector
    current_unit = st.selectbox(
        "Select Test Turbofan Unit",
        options=available_units,
        index=available_units.index(st.session_state.selected_unit)
        if st.session_state.selected_unit in available_units
        else 0,
        help="Select any of the 100 test turbofan engines from the CMAPSS FD001 dataset.",
    )

    if current_unit != st.session_state.selected_unit:
        st.session_state.selected_unit = current_unit
        st.session_state.current_cycle = 1
        st.session_state.autoplay = False

    engine_data = test_df[test_df["unit_number"] == current_unit].sort_values("time_cycles")
    max_cycles = int(engine_data["time_cycles"].max())
    ground_truth_final_rul = float(engine_data["RUL"].iloc[-1])
    total_life = max_cycles + ground_truth_final_rul

    # Quick Engine Stats
    st.markdown(
        f"""
        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: -6px; margin-bottom: 12px; padding: 8px; background: rgba(30,41,59,0.5); border-radius: 6px;">
            <div><b>Recorded Cycles:</b> {max_cycles} cycles</div>
            <div><b>Final True RUL:</b> {ground_truth_final_rul:.0f} cycles</div>
            <div><b>Estimated Lifespan:</b> {total_life:.0f} cycles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Cycle Scrubber Slider
    if st.session_state.current_cycle > max_cycles:
        st.session_state.current_cycle = max_cycles
    if st.session_state.current_cycle < 1:
        st.session_state.current_cycle = 1

    selected_cycle = st.slider(
        "Operating Cycle Scrubber",
        min_value=1,
        max_value=max_cycles,
        value=st.session_state.current_cycle,
        step=1,
        help="Scrub through engine life. Each step sends the sensor window to the live API.",
    )
    st.session_state.current_cycle = selected_cycle

    # 3. Autoplay Controls
    st.markdown("#### ⏯️ Playback Controls")
    col_play, col_pause, col_reset = st.columns(3)

    with col_play:
        if st.button("▶ Play", use_container_width=True):
            st.session_state.autoplay = True
    with col_pause:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.autoplay = False
    with col_reset:
        if st.button("⏮ Reset", use_container_width=True):
            st.session_state.current_cycle = 1
            st.session_state.autoplay = False
            st.rerun()

    play_speed = st.select_slider(
        "Playback Interval",
        options=[0.1, 0.2, 0.5, 1.0],
        value=0.5,
        format_func=lambda x: f"{x}s/cycle",
    )

    # 4. Model Selection Toggle
    st.markdown("#### 🧠 Model Visualization")
    model_toggle = st.radio(
        "Compare Models",
        options=["Both (Overlay)", "LSTM (Live API)", "XGBoost (Baseline)"],
        index=0,
        help="Toggle overlay to observe calibration differences as engines degrade toward failure.",
    )

    # 5. Pre-warm / Cache Full Trajectory Button
    st.markdown("#### ⚡ Performance")
    if st.button("🚀 Pre-fetch Trajectory", use_container_width=True, help="Pre-fetches all cycles for this engine into cache for lag-free scrubbing."):
        progress_bar = st.progress(0, text="Fetching live predictions...")
        step = max(1, max_cycles // 50)
        cycles_to_fetch = list(range(1, max_cycles + 1, step))
        if max_cycles not in cycles_to_fetch:
            cycles_to_fetch.append(max_cycles)

        for idx, c in enumerate(cycles_to_fetch):
            readings = build_sensor_payload(engine_data, c, max_cycles=30)
            payload_str = json.dumps({"readings": readings})
            query_live_api(st.session_state.api_url, current_unit, c, payload_str)
            progress_bar.progress((idx + 1) / len(cycles_to_fetch), text=f"Cached cycle {c}/{max_cycles}...")
        progress_bar.empty()
        st.success(f"Trajectory pre-fetched for Unit {current_unit}!")

    # 6. Live API Health & Connection
    st.markdown("---")
    st.markdown("#### 🌐 Live API Connection")
    api_url_input = st.text_input("FastAPI Endpoint", value=st.session_state.api_url)
    if api_url_input != st.session_state.api_url:
        st.session_state.api_url = api_url_input
        st.session_state.last_health_check = 0

    # Probe health periodically or on demand
    now = time.time()
    if st.session_state.api_status_cache is None or (now - st.session_state.last_health_check > 45):
        st.session_state.api_status_cache = check_api_health(st.session_state.api_url, timeout=4.0)
        st.session_state.last_health_check = now

    api_health = st.session_state.api_status_cache

    if api_health["online"]:
        st.markdown(
            f'<div class="badge-online">● LIVE ({api_health["model_version"]})</div>',
            unsafe_allow_html=True,
        )
    elif api_health["status"] == "sleeping":
        st.markdown(
            '<div class="badge-sleeping">● RENDER FREE TIER SLEEPING</div>',
            unsafe_allow_html=True,
        )
        if st.button("⏰ Wake Up API", use_container_width=True):
            with st.spinner("Waking up Render container (takes 30-50s)..."):
                st.session_state.api_status_cache = check_api_health(st.session_state.api_url, timeout=50.0)
                st.session_state.last_health_check = time.time()
                st.rerun()
    else:
        st.markdown(
            f'<div class="badge-error">● OFFLINE ({api_health["status"]})</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{api_health['error']}")


# ---------------------------------------------------------------------------
# Main Panel: Header & Cold Start Alert
# ---------------------------------------------------------------------------
col_head, col_badge = st.columns([4, 1])
with col_head:
    st.markdown("# ✈️ CMAPSS Turbofan RUL Prediction")
    st.markdown(
        "**Real-time model calibration & degradation visualizer.** Scrub through operating cycles of NASA's CMAPSS "
        "turbofan fleet, watch remaining useful life decay toward failure, and visually examine how **LSTM** avoids "
        "late-stage overconfident errors that baseline **XGBoost** models exhibit."
    )

with col_badge:
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    if api_health["online"]:
        st.markdown(
            f'<div class="badge-online" style="font-size: 0.85rem; padding: 6px 14px;">● Connected to Live API<br><span style="font-size: 0.7rem; color:#64748B;">{api_health["model_version"]}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="badge-sleeping" style="font-size: 0.85rem; padding: 6px 14px;">● Render Service Asleep<br><span style="font-size: 0.7rem; color:#64748B;">Spinning up on request</span></div>',
            unsafe_allow_html=True,
        )

# Render Cold-Start notification banner
if not api_health["online"] and api_health["status"] == "sleeping":
    st.info(
        "ℹ️ **Render Free-Tier Notice**: The live API spins down during idle periods. "
        "The first live prediction request may take 30–50 seconds while the container boots up. "
        "Subsequent requests respond in ~200ms.",
        icon="⏳",
    )

# ---------------------------------------------------------------------------
# Current Cycle Data & Real-Time Live API Inference
# ---------------------------------------------------------------------------
current_cycle_row = engine_data[engine_data["time_cycles"] == selected_cycle].iloc[0]
true_rul = float(current_cycle_row["RUL"])

# Extract sensor readings payload for the live API
readings_payload = build_sensor_payload(engine_data, selected_cycle, max_cycles=30)
payload_json_str = json.dumps({"readings": readings_payload})

# Query live API (cached per unit + cycle)
with st.spinner("Querying live model service..." if not api_health["online"] else None):
    api_result = query_live_api(
        st.session_state.api_url,
        current_unit,
        selected_cycle,
        payload_json_str,
    )

lstm_pred = api_result["predicted_rul"] if api_result["success"] else None

# Precomputed Stage 3 XGBoost prediction
unit_str = str(current_unit)
cycle_str = str(selected_cycle)
xgb_pred = xgb_preds_all.get(unit_str, {}).get(cycle_str, None)

# ---------------------------------------------------------------------------
# Section 1: KPI Telemetry Cards
# ---------------------------------------------------------------------------
kpi_cols = st.columns(5)

with kpi_cols[0]:
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">Current Cycle</div>
            <div class="kpi-value" style="color: #F8FAFC;">{selected_cycle} <span style="font-size: 1rem; color: #64748B;">/ {max_cycles}</span></div>
            <div class="kpi-sub">{(selected_cycle/max_cycles)*100:.1f}% of recorded run</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">True Ground RUL</div>
            <div class="kpi-value" style="color: #10B981;">{true_rul:.0f} <span style="font-size: 0.9rem;">cycles</span></div>
            <div class="kpi-sub">Actual failure at cycle {total_life:.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    lstm_display = f"{lstm_pred:.1f}" if lstm_pred is not None else "N/A"
    lstm_color = "#38BDF8" if lstm_pred is not None else "#64748B"
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">LSTM Predicted RUL</div>
            <div class="kpi-value" style="color: {lstm_color};">{lstm_display} <span style="font-size: 0.9rem;">cycles</span></div>
            <div class="kpi-sub">Live API • 30-cycle sequence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    xgb_display = f"{xgb_pred:.1f}" if xgb_pred is not None else "N/A"
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">XGBoost Baseline</div>
            <div class="kpi-value" style="color: #F59E0B;">{xgb_display} <span style="font-size: 0.9rem;">cycles</span></div>
            <div class="kpi-sub">Stage 3 • 101 flat features</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[4]:
    # Error delta: d = pred - actual.
    # Positive d = optimistic (overconfident/dangerous, penalized more heavily)
    # Negative d = conservative (early warning, penalized less)
    primary_pred = lstm_pred if lstm_pred is not None else xgb_pred
    if primary_pred is not None:
        delta = primary_pred - true_rul
        delta_sign = "+" if delta > 0 else ""
        delta_color = "#F43F5E" if delta > 15 else ("#F59E0B" if delta > 0 else "#10B981")
        penalty_desc = "Early Warning (Safe)" if delta <= 0 else "Optimistic (Penalized)"
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-title">Prediction Error (Δ)</div>
                <div class="kpi-value" style="color: {delta_color};">{delta_sign}{delta:.1f} <span style="font-size: 0.9rem;">cycles</span></div>
                <div class="kpi-sub">{penalty_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="kpi-container">
                <div class="kpi-title">Prediction Error (Δ)</div>
                <div class="kpi-value" style="color: #64748B;">--</div>
                <div class="kpi-sub">Awaiting inference</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Section 1 (Centerpiece): Interactive RUL Trajectory Comparison Chart
# ---------------------------------------------------------------------------
st.markdown("### 📈 RUL Degradation Trajectory")

# Compile historical curves up to max_cycles (with highlighted active scrub point)
all_cycles = engine_data["time_cycles"].values
true_rul_all = engine_data["RUL"].values

# Fetch cached LSTM predictions across all cycles for smooth curves
lstm_rul_curve = []
for c in all_cycles:
    # Check cache directly via query_live_api
    payload = build_sensor_payload(engine_data, int(c), max_cycles=30)
    p_str = json.dumps({"readings": payload})
    res = query_live_api(st.session_state.api_url, current_unit, int(c), p_str)
    if res["success"]:
        lstm_rul_curve.append(res["predicted_rul"])
    else:
        lstm_rul_curve.append(None)

# Precomputed XGBoost curve
xgb_rul_curve = [
    xgb_preds_all.get(unit_str, {}).get(str(int(c)), None) for c in all_cycles
]

# Build Plotly Figure
fig_rul = go.Figure()

# 1. Ground Truth RUL (Solid Emerald)
fig_rul.add_trace(
    go.Scatter(
        x=all_cycles,
        y=true_rul_all,
        mode="lines",
        name="Ground Truth RUL",
        line=dict(color="#10B981", width=3.5),
        hovertemplate="Cycle: %{x}<br>True RUL: %{y:.1f} cycles<extra></extra>",
    )
)

# 2. LSTM Prediction (Dashed Cyan)
if model_toggle in ["Both (Overlay)", "LSTM (Live API)"]:
    fig_rul.add_trace(
        go.Scatter(
            x=all_cycles,
            y=lstm_rul_curve,
            mode="lines",
            name="LSTM (Live API)",
            line=dict(color="#38BDF8", width=2.5, dash="dash"),
            hovertemplate="Cycle: %{x}<br>LSTM Pred: %{y:.1f} cycles<extra></extra>",
        )
    )

# 3. XGBoost Baseline (Dashed Amber)
if model_toggle in ["Both (Overlay)", "XGBoost (Baseline)"]:
    fig_rul.add_trace(
        go.Scatter(
            x=all_cycles,
            y=xgb_rul_curve,
            mode="lines",
            name="XGBoost Baseline",
            line=dict(color="#F59E0B", width=2.5, dash="dot"),
            hovertemplate="Cycle: %{x}<br>XGBoost Pred: %{y:.1f} cycles<extra></extra>",
        )
    )

# 4. Vertical Cursor Line for Current Scrubbed Position
fig_rul.add_vline(
    x=selected_cycle,
    line_width=2,
    line_dash="dash",
    line_color="#E2E8F0",
    annotation_text=f"Cycle {selected_cycle}",
    annotation_position="top left",
    annotation_font_color="#E2E8F0",
)

# 5. Marker Points at the Current Position
fig_rul.add_trace(
    go.Scatter(
        x=[selected_cycle],
        y=[true_rul],
        mode="markers",
        name="Current True RUL",
        marker=dict(color="#10B981", size=11, symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
        showlegend=False,
    )
)

if lstm_pred is not None and model_toggle in ["Both (Overlay)", "LSTM (Live API)"]:
    fig_rul.add_trace(
        go.Scatter(
            x=[selected_cycle],
            y=[lstm_pred],
            mode="markers",
            name="Current LSTM Pred",
            marker=dict(color="#38BDF8", size=11, symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
            showlegend=False,
        )
    )

if xgb_pred is not None and model_toggle in ["Both (Overlay)", "XGBoost (Baseline)"]:
    fig_rul.add_trace(
        go.Scatter(
            x=[selected_cycle],
            y=[xgb_pred],
            mode="markers",
            name="Current XGB Pred",
            marker=dict(color="#F59E0B", size=11, symbol="square", line=dict(color="#FFFFFF", width=1.5)),
            showlegend=False,
        )
    )

# Visual Callout annotation for late-stage divergence
if max_cycles > 50:
    late_cycle = max_cycles - 15
    fig_rul.add_annotation(
        x=late_cycle,
        y=true_rul_all[all_cycles == late_cycle][0] if len(all_cycles[all_cycles == late_cycle]) > 0 else 20,
        text="Critical End-of-Life Zone<br>LSTM avoids XGBoost overshoot",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#94A3B8",
        ax=0,
        ay=-45,
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="#38BDF8",
        font=dict(size=11, color="#E2E8F0"),
    )

fig_rul.update_layout(
    template="plotly_dark",
    height=420,
    margin=dict(l=40, r=30, t=30, b=40),
    xaxis=dict(
        title="Operating Cycles",
        gridcolor="rgba(148, 163, 184, 0.1)",
        showgrid=True,
    ),
    yaxis=dict(
        title="Remaining Useful Life (Cycles)",
        gridcolor="rgba(148, 163, 184, 0.1)",
        showgrid=True,
        rangemode="tozero",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(15, 23, 42, 0.7)",
        bordercolor="rgba(148, 163, 184, 0.2)",
    ),
    hovermode="x unified",
)

st.plotly_chart(fig_rul, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2: Informative Sensor Telemetry Trajectories (2x3 Grid)
# ---------------------------------------------------------------------------
st.markdown("### 🔬 Informative Sensor Trajectories")
st.caption(
    "Monitoring the 6 key degradation sensors identified during Stage 1 & 2 exploratory data analysis. "
    "Values are tracked from cycle 1 up to the current scrubbed cycle."
)

history_df = engine_data[engine_data["time_cycles"] <= selected_cycle]

sensor_grid_cols = st.columns(3)

for idx, sensor_key in enumerate(INFORMATIVE_SENSORS):
    col_target = sensor_grid_cols[idx % 3]
    meta = SENSOR_METADATA[sensor_key]

    curr_val = float(current_cycle_row[sensor_key])
    init_val = float(engine_data[sensor_key].iloc[0])
    delta_val = curr_val - init_val

    with col_target:
        # Micro Sparkline/Line Chart for Sensor
        fig_sensor = go.Figure()
        fig_sensor.add_trace(
            go.Scatter(
                x=history_df["time_cycles"],
                y=history_df[sensor_key],
                mode="lines",
                name=sensor_key,
                line=dict(color=meta["color"], width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(bytes.fromhex(meta['color'].lstrip('#'))) + [25])}",
                hovertemplate="Cycle %{x}: %{y:.2f} " + meta["unit"] + "<extra></extra>",
            )
        )

        fig_sensor.update_layout(
            template="plotly_dark",
            height=180,
            margin=dict(l=30, r=20, t=10, b=30),
            xaxis=dict(
                title=None,
                showgrid=False,
                range=[1, max_cycles],
            ),
            yaxis=dict(
                title=None,
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.08)",
            ),
            showlegend=False,
        )

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #E2E8F0;">{meta['label']}</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 4px 0 8px 0;">
                    <span style="font-size: 1.4rem; font-weight: 700; color: {meta['color']};">{curr_val:.2f} <span style="font-size: 0.8rem; color: #94A3B8;">{meta['unit']}</span></span>
                    <span style="font-size: 0.8rem; color: {'#10B981' if abs(delta_val) < 1e-2 else '#F43F5E' if delta_val > 0 else '#38BDF8'};">
                        {'+' if delta_val > 0 else ''}{delta_val:.2f} from start
                    </span>
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; line-height: 1.3; min-height: 38px;">{meta['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_sensor, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Model Benchmark Comparison Panel (experiments.json)
# ---------------------------------------------------------------------------
st.markdown("### 🏆 Stage 3 Model Comparison & Metric Benchmark")
st.markdown(
    "While XGBoost and LSTM achieve near-identical RMSE (~13.8) and MAE (~10.1), **LSTM reduces the CMAPSS asymmetric penalty score by 31%** "
    "(12,253 vs 17,865). In predictive maintenance, falsely predicting that an engine has life remaining when it is about to fail is far more "
    "hazardous than scheduling early maintenance."
)

comp_col_table, comp_col_chart = st.columns([1, 1])

# Extract experiment metrics
exp_df = pd.DataFrame(experiments_data)
# Format display dataframe
display_metrics = []
for item in experiments_data:
    display_metrics.append({
        "Model": item["model"],
        "Val RMSE": f"{item['val_rmse']:.2f}",
        "Val MAE": f"{item['val_mae']:.2f}",
        "CMAPSS Score": f"{item['val_cmapss_score']:,.0f}",
        "Training Time": f"{item['training_time_s']:.1f} s",
    })

with comp_col_table:
    st.markdown("#### Model Performance Summary")
    st.dataframe(
        pd.DataFrame(display_metrics),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        """
        <div class="callout-box">
            <b>Why CMAPSS Score Decided the Production Model:</b><br>
            NASA's asymmetric penalty uses exponential weight:
            <code>s = Σ (e^(d/10) - 1)</code> for optimistic errors (d > 0), whereas late predictions use <code>e^(-d/13) - 1</code>.
            Because LSTM handles sequential dependencies across consecutive cycles, it avoids dangerous late-life overshoots.
        </div>
        """,
        unsafe_allow_html=True,
    )

with comp_col_chart:
    st.markdown("#### Metric Comparison (CMAPSS Penalty vs RMSE)")

    models = [item["model"] for item in experiments_data]
    cmapss_scores = [item["val_cmapss_score"] for item in experiments_data]
    rmses = [item["val_rmse"] for item in experiments_data]

    fig_comp = make_subplots(specs=[[{"secondary_y": True}]])

    fig_comp.add_trace(
        go.Bar(
            x=models,
            y=cmapss_scores,
            name="CMAPSS Asymmetric Score (Lower = Better)",
            marker_color=["#F59E0B", "#10B981"],
            text=[f"{v:,.0f}" for v in cmapss_scores],
            textposition="auto",
        ),
        secondary_y=False,
    )

    fig_comp.add_trace(
        go.Scatter(
            x=models,
            y=rmses,
            name="Val RMSE",
            mode="lines+markers+text",
            marker=dict(size=10, color="#38BDF8"),
            line=dict(width=2, dash="dash", color="#38BDF8"),
            text=[f"{v:.2f}" for v in rmses],
            textposition="top center",
        ),
        secondary_y=True,
    )

    fig_comp.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="CMAPSS Score", showgrid=False),
        yaxis2=dict(title="RMSE", showgrid=False, range=[10, 18]),
    )

    st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------------------------
# Autoplay Stepper Loop
# ---------------------------------------------------------------------------
if st.session_state.autoplay:
    if st.session_state.current_cycle < max_cycles:
        time.sleep(play_speed)
        st.session_state.current_cycle += 1
        st.rerun()
    else:
        st.session_state.autoplay = False
        st.toast(f"Engine {current_unit} run completed!", icon="✅")

# ---------------------------------------------------------------------------
# Footer & About Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-container">
        <b>NASA CMAPSS Turbofan Fleet Predictive Maintenance</b><br>
        Built with Streamlit • FastAPI Serving Layer on Render • PyTorch LSTM • XGBoost Baseline<br>
        <a href="https://github.com/AaruneshAP/Project1_cat" target="_blank" style="color: #38BDF8; text-decoration: none; margin-right: 16px;">GitHub Repository</a>
        <a href="https://cmapss-rul-api-gfxh.onrender.com/docs" target="_blank" style="color: #38BDF8; text-decoration: none; margin-right: 16px;">Interactive API Docs (Swagger)</a>
        <a href="https://cmapss-rul-api-gfxh.onrender.com/health" target="_blank" style="color: #38BDF8; text-decoration: none;">Service Health Check</a>
    </div>
    """,
    unsafe_allow_html=True,
)
