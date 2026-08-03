# Feature Engineering Notes — CMAPSS FD001

## Feature Selection Philosophy
The goal is to transform raw sensor time series into features that capture:
1. **Current state** — the raw sensor value at each cycle
2. **Smoothed state** — rolling mean to reduce noise
3. **Instability** — rolling std to capture signal variability
4. **Degradation trend** — rolling slope to detect accelerating wear

## Dropped Sensors
7 sensors removed due to near-zero variance (std < 0.01 across entire training set):
- sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19
- These sensors provide constant readings regardless of engine health

## Retained Sensors (14)
sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21

### Physical Justification
- **sensor_2 (T2)**: Total temperature at fan inlet — increases with degradation as compressor efficiency drops
- **sensor_3 (T3)**: Total temperature at HPC outlet — direct indicator of compressor health
- **sensor_4 (T4)**: Total temperature at LPT outlet — reflects turbine degradation
- **sensor_7 (Ps30)**: Static pressure at HPC outlet — changes with flow path geometry degradation
- **sensor_8 (phi)**: Ratio of fuel flow to Ps30 — efficiency proxy
- **sensor_9 (NRf)**: Corrected fan speed — compensated for ambient conditions
- **sensor_11 (T50)**: Total temperature at HPT outlet — key turbine health indicator
- **sensor_14 (NRc)**: Corrected core speed — highly correlated with sensor_9 (r=0.963)
- **sensor_15, 17, 20, 21**: Various bypass/bleed flow measurements that shift with degradation

## High Correlation Pairs
- **sensor_9 <-> sensor_14**: r = 0.9632
  - Both are corrected rotational speeds (fan vs. core). Kept both because the slight decorrelation during degradation may carry diagnostic value. A production system might drop one.

## Rolling Window Features
For each of the 14 active sensors, computed with window sizes 5 and 20:
- **Rolling mean**: Reduces cycle-to-cycle noise, captures the underlying trend
- **Rolling std**: Captures signal instability — engines often show increased sensor variance before failure
- **Rolling slope**: Linear regression coefficient over the window — a direct measure of degradation rate. Negative slopes in temperature sensors indicate cooling (good), positive slopes indicate heating (degradation).

Total: 14 sensors × 2 windows × 3 stats = 84 rolling features + 14 raw sensors + 3 settings = **101 features**

## RUL Clipping
Training RUL clipped at 125 cycles (standard CMAPSS practice).

**Rationale**: In the early life of an engine (say, at cycle 10 of a 350-cycle life), sensor readings are indistinguishable from an engine at cycle 10 of a 200-cycle life. The degradation hasn't manifested in the sensor data yet. Clipping forces the model to focus on the last ~125 cycles where degradation is actually observable, making the regression target more learnable.

## Scaling
StandardScaler (zero mean, unit variance) fitted on training data only. Test data transformed with the same scaler to prevent data leakage.
