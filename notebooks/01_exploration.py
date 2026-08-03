"""
Stage 1 — Data Acquisition & Exploration
==========================================
CMAPSS Turbofan Engine Degradation Simulation Data Set (FD001)

This script:
  1. Loads train_FD001.txt, test_FD001.txt, RUL_FD001.txt
  2. Assigns standard column names
  3. Performs quality checks (nulls, variance)
  4. Constructs training RUL labels
  5. Produces exploratory plots
  6. Saves cleaned/labeled data as parquet files

Dataset-agnostic where possible — column naming and loading logic work for
FD001–FD004 without modification.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cmapss_extracted")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
SUBSET = "FD001"  # Change to FD002–FD004 to process other subsets

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Column Definitions
# ---------------------------------------------------------------------------
# Standard CMAPSS column names (26 columns total in training/test files):
#   unit_number, time_cycles, 3 operational settings, 21 sensors
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = ["unit_number", "time_cycles"] + SETTING_COLS + SENSOR_COLS


def load_data(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load train, test, and RUL files for a given CMAPSS subset."""
    train_path = os.path.join(DATA_DIR, f"train_{subset}.txt")
    test_path = os.path.join(DATA_DIR, f"test_{subset}.txt")
    rul_path = os.path.join(DATA_DIR, f"RUL_{subset}.txt")

    # Files are space-delimited with trailing spaces → extra empty columns
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, engine="python")
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, engine="python")
    rul_series = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python").iloc[:, 0]

    # Drop any extra columns (trailing whitespace creates phantom columns)
    train_df = train_df.iloc[:, :len(COLUMN_NAMES)]
    test_df = test_df.iloc[:, :len(COLUMN_NAMES)]

    train_df.columns = COLUMN_NAMES
    test_df.columns = COLUMN_NAMES

    print(f"[{subset}] Train shape: {train_df.shape}")
    print(f"[{subset}] Test  shape: {test_df.shape}")
    print(f"[{subset}] RUL entries: {len(rul_series)}")

    return train_df, test_df, rul_series


def check_nulls(df: pd.DataFrame, label: str) -> None:
    """Check and report null values. CMAPSS is expected to be clean."""
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"\n--- Null Check ({label}) ---")
    if total_nulls == 0:
        print("[OK] No null values found (as expected for CMAPSS).")
    else:
        print(f"[WARN] Found {total_nulls} null values!")
        print(null_counts[null_counts > 0])


def identify_low_variance_sensors(df: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    """
    Flag sensors with near-zero standard deviation across the full training set.
    
    A sensor with std ≈ 0 provides no discriminating information for RUL
    prediction — it reads essentially the same value regardless of engine
    health state.
    
    Returns list of low-variance sensor column names.
    """
    sensor_stds = df[SENSOR_COLS].std().sort_values()
    print("\n--- Sensor Standard Deviations ---")
    print(sensor_stds.to_string())

    low_var = sensor_stds[sensor_stds < threshold].index.tolist()
    print(f"\nLow-variance sensors (std < {threshold}): {low_var}")
    print("These sensors will be candidates for removal in Stage 2.")
    return low_var


def construct_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct Remaining Useful Life for each row in the training set.
    
    Formula: RUL = max_cycle_for_unit - current_time_cycle
    
    At cycle 1 of a 200-cycle engine, RUL = 199.
    At the last cycle, RUL = 0.
    """
    max_cycles = train_df.groupby("unit_number")["time_cycles"].max()
    train_df = train_df.copy()
    train_df["RUL"] = train_df.apply(
        lambda row: max_cycles[row["unit_number"]] - row["time_cycles"], axis=1
    )
    print(f"\nRUL constructed. Range: [{train_df['RUL'].min()}, {train_df['RUL'].max()}]")
    return train_df


def attach_test_rul(test_df: pd.DataFrame, rul_series: pd.Series) -> pd.DataFrame:
    """
    Attach ground-truth RUL to the test set.
    
    The RUL file gives the remaining life at the LAST cycle of each test engine.
    For earlier cycles: RUL_at_cycle = RUL_at_last_cycle + (max_cycle - current_cycle)
    """
    test_df = test_df.copy()
    max_cycles = test_df.groupby("unit_number")["time_cycles"].max()

    # Build a unit_number → ground truth RUL mapping
    unit_ids = test_df["unit_number"].unique()
    rul_map = dict(zip(unit_ids, rul_series.values))

    # For each row, compute full RUL
    test_df["RUL"] = test_df.apply(
        lambda row: rul_map[row["unit_number"]] + (
            max_cycles[row["unit_number"]] - row["time_cycles"]
        ),
        axis=1,
    )
    return test_df


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------
def plot_sensor_trajectories(df: pd.DataFrame, n_units: int = 5) -> None:
    """Plot sensor readings over time for a sample of engines."""
    sample_units = sorted(df["unit_number"].unique())[:n_units]
    active_sensors = [s for s in SENSOR_COLS if df[s].std() > 0.01]

    fig, axes = plt.subplots(
        len(active_sensors), 1,
        figsize=(14, 3 * len(active_sensors)),
        sharex=True,
    )
    if len(active_sensors) == 1:
        axes = [axes]

    for ax, sensor in zip(axes, active_sensors):
        for unit in sample_units:
            unit_data = df[df["unit_number"] == unit]
            ax.plot(unit_data["time_cycles"], unit_data[sensor], alpha=0.7, label=f"Unit {unit}")
        ax.set_ylabel(sensor, fontsize=8)
        ax.tick_params(labelsize=7)
    
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time Cycles")
    fig.suptitle(f"Sensor Trajectories — {n_units} Sample Engines", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sensor_trajectories.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, 'sensor_trajectories.png')}")


def plot_lifespan_histogram(df: pd.DataFrame) -> None:
    """Histogram of engine lifespans (max cycle per unit)."""
    lifespans = df.groupby("unit_number")["time_cycles"].max()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lifespans, bins=30, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.set_xlabel("Engine Lifespan (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Engine Lifespans")
    ax.axvline(lifespans.mean(), color="red", linestyle="--", label=f"Mean: {lifespans.mean():.0f}")
    ax.axvline(lifespans.median(), color="orange", linestyle="--", label=f"Median: {lifespans.median():.0f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lifespan_histogram.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, 'lifespan_histogram.png')}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap of sensor readings."""
    corr = df[SENSOR_COLS].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Sensor Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, 'correlation_heatmap.png')}")


def plot_rul_distribution(df: pd.DataFrame) -> None:
    """Distribution of RUL values in training data."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["RUL"], bins=50, edgecolor="black", alpha=0.75, color="#55A868")
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of RUL in Training Data")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rul_distribution.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, 'rul_distribution.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"CMAPSS Data Exploration — Subset {SUBSET}")
    print("=" * 60)

    # 1. Load data
    train_df, test_df, rul_series = load_data(SUBSET)

    # 2. Null checks
    check_nulls(train_df, "Train")
    check_nulls(test_df, "Test")

    # 3. Low-variance sensor identification
    low_var_sensors = identify_low_variance_sensors(train_df)

    # 4. Lifespan analysis
    lifespans = train_df.groupby("unit_number")["time_cycles"].max()
    print(f"\n--- Engine Lifespan Stats ---")
    print(f"  Min:    {lifespans.min()}")
    print(f"  Max:    {lifespans.max()}")
    print(f"  Mean:   {lifespans.mean():.1f}")
    print(f"  Median: {lifespans.median():.1f}")
    print(f"  Std:    {lifespans.std():.1f}")

    # 5. Construct RUL
    train_df = construct_rul(train_df)
    test_df = attach_test_rul(test_df, rul_series)

    # 6. Generate plots
    print("\n--- Generating Plots ---")
    plot_sensor_trajectories(train_df)
    plot_lifespan_histogram(train_df)
    plot_correlation_heatmap(train_df)
    plot_rul_distribution(train_df)

    # 7. Save parquets
    train_out = os.path.join(OUTPUT_DIR, "train_fd001.parquet")
    test_out = os.path.join(OUTPUT_DIR, "test_fd001.parquet")
    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)
    print(f"\nSaved: {train_out}")
    print(f"Saved: {test_out}")

    # 8. Save exploration metadata for downstream stages
    exploration_meta = {
        "subset": SUBSET,
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "n_units_train": int(train_df["unit_number"].nunique()),
        "n_units_test": int(test_df["unit_number"].nunique()),
        "low_variance_sensors": low_var_sensors,
        "sensor_stds": train_df[SENSOR_COLS].std().to_dict(),
        "lifespan_stats": {
            "min": int(lifespans.min()),
            "max": int(lifespans.max()),
            "mean": float(lifespans.mean()),
            "median": float(lifespans.median()),
        },
    }
    meta_path = os.path.join(OUTPUT_DIR, "exploration_meta.json")
    with open(meta_path, "w") as f:
        json.dump(exploration_meta, f, indent=2)
    print(f"Saved: {meta_path}")

    print("\n[DONE] Stage 1 complete.")
    return exploration_meta


if __name__ == "__main__":
    meta = main()
