# EDA Notes — CMAPSS FD001 Exploration Findings

## Dataset Overview
- **Training set**: 20,631 rows × 26 columns, covering 100 engines
- **Test set**: 13,096 rows × 26 columns, covering 100 engines
- **RUL file**: 100 ground-truth RUL values (one per test engine, at the last recorded cycle)
- **Null values**: None found (CMAPSS is a synthetic, clean dataset — confirmed, not assumed)

## Engine Lifespans
- Range: 128 to 362 cycles
- Mean: 206.3 cycles
- Median: 199.0 cycles
- Std: 46.3 cycles
- The distribution is roughly bell-shaped but with a slight right skew — some engines last significantly longer than average.

## Sensor Analysis

### Dropped Sensors (Near-Zero Variance)
The following 7 sensors were flagged for removal due to standard deviation < 0.01:

| Sensor     | Std Dev       | Reason for Drop                                      |
|------------|---------------|------------------------------------------------------|
| sensor_1   | 0.000000      | Constant value — no information content              |
| sensor_5   | ~5.3e-15      | Effectively constant (floating point noise only)     |
| sensor_6   | 0.001389      | Negligible variation across all engines              |
| sensor_10  | 0.000000      | Constant value — no information content              |
| sensor_16  | ~3.5e-18      | Effectively constant (floating point noise only)     |
| sensor_18  | 0.000000      | Constant value — no information content              |
| sensor_19  | 0.000000      | Constant value — no information content              |

These sensors read essentially the same value regardless of engine health state, providing no discriminating information for RUL prediction.

### Retained Sensors (14 total)
sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21

### Degradation Shape Observed
From the sensor trajectory plots:
- Most sensors show a **gradual monotonic trend** (increasing or decreasing) as engines degrade
- The degradation is not linear — it often **accelerates** in the final ~50 cycles
- sensor_2, sensor_3, sensor_4, sensor_11 show the clearest degradation trends
- sensor_9 and sensor_14 are highly correlated (r = 0.96), suggesting they measure related physical quantities
- sensor_7, sensor_15, and sensor_21 show more noisy but still informative trends
- sensor_8 and sensor_13 have relatively small dynamic ranges but still carry degradation signal

### Operational Settings
- setting_1, setting_2: Vary between engines/cycles (operational conditions)
- setting_3: Near-constant in FD001 (this is by design — FD001 is a single operating condition subset)
