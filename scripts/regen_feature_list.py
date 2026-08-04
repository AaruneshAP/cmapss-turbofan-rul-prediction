"""
Regenerate feature_list.json from the LSTM checkpoint.

Run from the project root:
    python scripts/regen_feature_list.py
"""

import torch
import json
import os

ARTIFACTS = os.path.join("serving", "model_artifacts")

# ── 1. Load checkpoint ────────────────────────────────────────────────────────
ckpt = torch.load(
    os.path.join(ARTIFACTS, "lstm_model.pt"),
    map_location="cpu",
    weights_only=False,
)

sensor_cols = ckpt["sensor_cols"]  # ground-truth list embedded at training time
input_size  = ckpt["input_size"]   # must equal len(sensor_cols)

# ── 2. Patch seq_length into checkpoint if absent ────────────────────────────
# model_selection.json records params.seq_length = 30 for the LSTM run.
SEQ_LENGTH = 30
if "seq_length" not in ckpt:
    ckpt["seq_length"] = SEQ_LENGTH
    torch.save(ckpt, os.path.join(ARTIFACTS, "lstm_model.pt"))
    print(f"Patched seq_length={SEQ_LENGTH} into lstm_model.pt")
else:
    SEQ_LENGTH = ckpt["seq_length"]
    print(f"seq_length already in checkpoint: {SEQ_LENGTH}")

# ── 3. Load old feature_list.json to preserve XGBoost feature list ───────────
old_path = os.path.join(ARTIFACTS, "feature_list.json")
with open(old_path) as f:
    old = json.load(f)

# ── 4. Build new feature_list.json ───────────────────────────────────────────
# For the LSTM the only columns used at inference are sensor_cols (settings +
# active sensors — no rolling aggregates; LSTM learns temporal patterns itself).
# feature_cols is intentionally the same as sensor_cols for the LSTM path.

dropped_sensors = [
    f"sensor_{i}" for i in range(1, 22)
    if f"sensor_{i}" not in sensor_cols
]

new_feature_list = {
    "_note": (
        "Regenerated from lstm_model.pt checkpoint on 2026-08-04. "
        "sensor_cols lists the 17 columns (3 settings + 14 active sensors) "
        "the LSTM uses as input features, in the exact order stored in the "
        "checkpoint. feature_cols_xgboost retains the original 99-column "
        "XGBoost flat-feature list for reference only."
    ),
    "model": "LSTM",
    "input_size": input_size,
    "seq_length": SEQ_LENGTH,
    # Authoritative column list — used by app.py at inference time
    "sensor_cols": sensor_cols,
    # feature_cols == sensor_cols for the LSTM (no rolling stats needed)
    "feature_cols": sensor_cols,
    # Preserve old XGBoost list for audit / potential future use
    "feature_cols_xgboost": old.get("feature_cols", []),
    # Sensors dropped in Stage 2 due to near-zero variance in FD001
    "dropped_sensors": dropped_sensors,
}

with open(old_path, "w") as f:
    json.dump(new_feature_list, f, indent=2)

print()
print("Written feature_list.json")
print(f"  input_size     : {input_size}")
print(f"  seq_length     : {SEQ_LENGTH}")
n = len(sensor_cols)
print(f"  sensor_cols    ({n}): {sensor_cols}")
nd = len(dropped_sensors)
print(f"  dropped_sensors({nd}): {dropped_sensors}")
