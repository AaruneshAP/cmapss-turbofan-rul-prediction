"""
Stage 3 — Modeling
===================
Trains and evaluates RUL prediction models on CMAPSS FD001 data.

Models:
  1. XGBoost Regressor (baseline) — trained on engineered flat features
  2. LSTM Neural Network (comparison) — trained on raw sensor sequences

Both models are evaluated with:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)  
  - CMAPSS Asymmetric Scoring Function (penalizes late predictions more)

The winning model is selected based on the CMAPSS score.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "serving", "model_artifacts")
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

RANDOM_STATE = 42
VAL_FRACTION = 0.2  # fraction of units held out for validation

# Whether to train LSTM (requires torch; skip if not available)
TRAIN_LSTM = True
LSTM_SEQUENCE_LENGTH = 30  # sliding window size for LSTM


# ---------------------------------------------------------------------------
# CMAPSS Asymmetric Scoring Function
# ---------------------------------------------------------------------------
def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    CMAPSS asymmetric scoring function.
    
    This is a key metric in the CMAPSS literature and a common interview topic.
    
    The function penalizes LATE predictions (predicted RUL < actual RUL, i.e.,
    the model says the engine will fail sooner than it actually does) LESS than
    EARLY predictions (predicted RUL > actual RUL, i.e., the model says the
    engine is healthier than it actually is).
    
    Rationale: In predictive maintenance, missing a failure (thinking the engine
    is fine when it's about to fail) is much more dangerous than being overly
    cautious (scheduling maintenance too early).
    
    Formula:
        d = predicted - actual (positive d means we predicted more life remaining)
        For d < 0 (late prediction / conservative): score += exp(-d/13) - 1
        For d >= 0 (early prediction / optimistic):  score += exp(d/10) - 1
        
    Note: "late prediction" means we predict failure will happen later than it 
    actually does (we're late to warn), which is penalized more heavily.
    """
    d = y_pred - y_true  # positive = optimistic (predicted more RUL than actual)
    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,  # conservative: smaller penalty
        np.exp(d / 10.0) - 1,   # optimistic: larger penalty
    )
    return float(np.sum(scores))


# ---------------------------------------------------------------------------
# Data Loading & Split
# ---------------------------------------------------------------------------
def load_features():
    """Load engineered features from Stage 2."""
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "features_train.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "features_test.parquet"))

    with open(os.path.join(ARTIFACTS_DIR, "feature_list.json")) as f:
        feat_info = json.load(f)

    feature_cols = feat_info["feature_cols"]
    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    return train_df, test_df, feature_cols


def split_by_unit(
    df: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by unit_number (never by row).
    
    Splitting by row would leak future cycles of the same engine into
    validation, giving an artificially inflated score. Splitting by
    unit ensures the model has never seen any data from validation engines.
    """
    rng = np.random.RandomState(seed)
    units = df["unit_number"].unique()
    rng.shuffle(units)

    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])

    train_split = df[df["unit_number"].isin(train_units)]
    val_split = df[df["unit_number"].isin(val_units)]

    print(f"\nTrain/Val split by unit_number:")
    print(f"  Train units: {len(train_units)} ({len(train_split)} rows)")
    print(f"  Val units:   {len(val_units)} ({len(val_split)} rows)")

    return train_split, val_split


# ---------------------------------------------------------------------------
# XGBoost Model
# ---------------------------------------------------------------------------
def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost regressor with hyperparameter tuning.
    
    Uses RandomizedSearchCV with GroupKFold (grouped by unit_number) to
    prevent leakage during cross-validation as well.
    """
    print("\n" + "=" * 60)
    print("Training XGBoost Regressor")
    print("=" * 60)

    X_train = train_df[feature_cols].values
    y_train = train_df["RUL"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["RUL"].values
    groups = train_df["unit_number"].values

    # Hyperparameter search space
    param_grid = {
        "max_depth": [3, 5, 7, 9],
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    }

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method="hist",  # fast histogram-based method
        verbosity=0,
    )

    # Use GroupKFold to respect unit boundaries during CV
    cv = GroupKFold(n_splits=3)
    
    print("Running RandomizedSearchCV (20 iterations, 3-fold GroupKFold)...")
    start_time = time.time()

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train, groups=groups)
    
    elapsed = time.time() - start_time
    print(f"Search completed in {elapsed:.1f}s")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # Evaluate on validation set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))
    score = cmapss_score(y_val, y_pred)

    print(f"\nValidation Metrics:")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  CMAPSS Score: {score:.2f}")

    results = {
        "model": "XGBoost",
        "params": search.best_params_,
        "cv_rmse": float(-search.best_score_),
        "val_rmse": rmse,
        "val_mae": mae,
        "val_cmapss_score": score,
        "training_time_s": elapsed,
    }

    return best_model, results


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------
def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_length: int = LSTM_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences for LSTM input.
    
    For each engine, we create sequences of `seq_length` consecutive cycles.
    Each sequence's target is the RUL at the last cycle in the window.
    
    Engines with fewer than `seq_length` cycles are padded (left-padded with
    the first cycle's values repeated).
    """
    X_sequences = []
    y_targets = []

    for unit_id in df["unit_number"].unique():
        unit_data = df[df["unit_number"] == unit_id].sort_values("time_cycles")
        features = unit_data[feature_cols].values
        rul_values = unit_data["RUL"].values

        # Pad short sequences
        if len(features) < seq_length:
            pad_length = seq_length - len(features)
            padding = np.repeat(features[:1], pad_length, axis=0)
            features = np.vstack([padding, features])
            rul_values = np.concatenate([np.repeat(rul_values[0], pad_length), rul_values])

        # Create sliding windows
        for i in range(seq_length, len(features) + 1):
            X_sequences.append(features[i - seq_length : i])
            y_targets.append(rul_values[i - 1])

    return np.array(X_sequences, dtype=np.float32), np.array(y_targets, dtype=np.float32)


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple:
    """
    Train LSTM model for RUL prediction.
    
    Uses a 2-layer LSTM with dropout, trained with Adam optimizer and
    MSE loss. Early stopping based on validation loss.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("\n[SKIP] PyTorch not installed. Skipping LSTM training.")
        print("  Install with: pip install torch")
        return None, None

    # --- All torch-dependent code is inside this block so that the names
    #     `torch`, `nn`, `TensorDataset`, and `DataLoader` are guaranteed to
    #     be defined before use and any unexpected torch error is surfaced. ---
    try:
        print("\n" + "=" * 60)
        print("Training LSTM Model")
        print("=" * 60)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Use only raw sensor columns (not rolling features) for LSTM
        # LSTM will learn temporal patterns itself
        sensor_cols = [c for c in feature_cols if "_w" not in c]
        print(f"LSTM input features: {len(sensor_cols)} (raw sensors + settings)")

        # Create sequences
        print("Creating sequences...")
        X_train, y_train = create_sequences(train_df, sensor_cols, LSTM_SEQUENCE_LENGTH)
        X_val, y_val = create_sequences(val_df, sensor_cols, LSTM_SEQUENCE_LENGTH)
        print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

        # PyTorch datasets
        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Model definition
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
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

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Use only the last time step's output
                last_output = lstm_out[:, -1, :]
                return self.fc(last_output).squeeze(-1)

        input_size = X_train.shape[2]
        model = LSTMModel(input_size=input_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop with early stopping
        n_epochs = 50
        patience = 10
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"\nTraining for up to {n_epochs} epochs (patience={patience})...")
        start_time = time.time()

        for epoch in range(n_epochs):
            # Train
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f}s")

        # Load best model and evaluate
        model.load_state_dict(best_state)
        model.eval()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(y_batch.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        score = cmapss_score(y_true, y_pred)

        print(f"\nValidation Metrics:")
        print(f"  RMSE:         {rmse:.4f}")
        print(f"  MAE:          {mae:.4f}")
        print(f"  CMAPSS Score: {score:.2f}")

        results = {
            "model": "LSTM",
            "params": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.3,
                "seq_length": LSTM_SEQUENCE_LENGTH,
                "batch_size": 256,
                "lr": 0.001,
                "epochs_trained": epoch + 1,
            },
            "val_rmse": rmse,
            "val_mae": mae,
            "val_cmapss_score": score,
            "training_time_s": elapsed,
        }

        # Save LSTM model
        model_path = os.path.join(ARTIFACTS_DIR, "lstm_model.pt")
        torch.save({
            "model_state_dict": best_state,
            "input_size": input_size,
            "sensor_cols": sensor_cols,
        }, model_path)
        print(f"Saved LSTM model: {model_path}")

        return model, results

    except Exception as e:
        print(f"\n[ERROR] LSTM training failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Experiment Logging
# ---------------------------------------------------------------------------
def log_experiment(results: dict, experiments_path: str) -> None:
    """Append experiment results to experiments.json."""
    if os.path.exists(experiments_path):
        with open(experiments_path) as f:
            experiments = json.load(f)
    else:
        experiments = []

    # Ensure all values are JSON-serializable
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean_results[k] = {
                kk: (int(vv) if isinstance(vv, np.integer) else
                     float(vv) if isinstance(vv, np.floating) else vv)
                for kk, vv in v.items()
            }
        elif isinstance(v, (np.integer,)):
            clean_results[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_results[k] = float(v)
        else:
            clean_results[k] = v

    experiments.append(clean_results)

    with open(experiments_path, "w") as f:
        json.dump(experiments, f, indent=2)
    print(f"Logged experiment to {experiments_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CMAPSS Modeling — Stage 3")
    print("=" * 60)

    experiments_path = os.path.join(PROJECT_DIR, "experiments.json")

    # 1. Load features
    train_df, test_df, feature_cols = load_features()

    # 2. Split by unit
    train_split, val_split = split_by_unit(train_df)

    # 3. Train XGBoost
    xgb_model, xgb_results = train_xgboost(train_split, val_split, feature_cols)
    log_experiment(xgb_results, experiments_path)

    # 4. Train LSTM (optional — requires PyTorch)
    lstm_results = None
    if TRAIN_LSTM:
        _, lstm_results = train_lstm(train_split, val_split, feature_cols)
        if lstm_results:
            log_experiment(lstm_results, experiments_path)

    # 5. Select winning model
    print("\n" + "=" * 60)
    print("Model Selection")
    print("=" * 60)

    candidates = [("XGBoost", xgb_results)]
    if lstm_results:
        candidates.append(("LSTM", lstm_results))

    print(f"\n{'Model':<12} {'RMSE':<10} {'MAE':<10} {'CMAPSS Score':<15}")
    print("-" * 47)
    for name, res in candidates:
        print(f"{name:<12} {res['val_rmse']:<10.4f} {res['val_mae']:<10.4f} {res['val_cmapss_score']:<15.2f}")

    # Select by CMAPSS score (lower is better)
    winner_name, winner_results = min(candidates, key=lambda x: x[1]["val_cmapss_score"])
    print(f"\nWinner: {winner_name} (CMAPSS Score: {winner_results['val_cmapss_score']:.2f})")

    # 6. Save winning model
    if winner_name == "XGBoost":
        model_path = os.path.join(ARTIFACTS_DIR, "model.json")
        xgb_model.save_model(model_path)
        print(f"Saved XGBoost model: {model_path}")

        # Also save as pkl for compatibility
        pkl_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
        joblib.dump(xgb_model, pkl_path)
        print(f"Saved XGBoost model (pkl): {pkl_path}")

    # Save model selection metadata
    selection_meta = {
        "selected_model": winner_name,
        "selection_criterion": "CMAPSS asymmetric score (lower is better)",
        "all_results": {name: res for name, res in candidates},
    }
    meta_path = os.path.join(ARTIFACTS_DIR, "model_selection.json")
    with open(meta_path, "w") as f:
        json.dump(selection_meta, f, indent=2)
    print(f"Saved: {meta_path}")

    # 7. Evaluate winner on test set (last cycle per unit only — standard eval)
    print("\n--- Test Set Evaluation (last cycle per unit) ---")
    test_last = test_df.groupby("unit_number").last().reset_index()
    
    if winner_name == "XGBoost":
        X_test = test_last[feature_cols].values
        y_test = test_last["RUL"].values
        y_pred_test = xgb_model.predict(X_test)
    else:
        print("  (LSTM test evaluation would use sequence data — see notebook for details)")
        # For simplicity in the script, we evaluate XGBoost on test too
        X_test = test_last[feature_cols].values
        y_test = test_last["RUL"].values
        y_pred_test = xgb_model.predict(X_test)

    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_score = cmapss_score(y_test, y_pred_test)

    print(f"  Test RMSE:         {test_rmse:.4f}")
    print(f"  Test MAE:          {test_mae:.4f}")
    print(f"  Test CMAPSS Score: {test_score:.2f}")

    print("\n[DONE] Stage 3 complete.")


if __name__ == "__main__":
    main()
