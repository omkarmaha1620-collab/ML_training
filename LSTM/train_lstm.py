import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = "LSTM/lstm_training_dataset.csv"

MODEL_FILE = "LSTM/lstm_model.keras"
SCALER_FILE = "LSTM/lstm_scaler.pkl"
RESULTS_FILE = "LSTM/lstm_results.csv"
PREDICTIONS_FILE = "LSTM/lstm_predictions.csv"

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("LSTM TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "region",
    "target_time",
    "target_VHM0"
]

for variable in ["VHM0", "VTPK", "VPED"]:
    for lag in range(1, 9):
        required_columns.append(f"{variable}_t-{lag}")

missing = [col for col in required_columns if col not in df.columns]

if missing:
    print("\nERROR: Missing columns:")
    for col in missing:
        print(col)
    raise ValueError("Required LSTM columns are missing.")


# ============================================================
# 4. PREPARE DATA
# ============================================================

print("\nPreparing data...")

df["target_time"] = pd.to_datetime(df["target_time"])

# Sort chronologically within each region
df = df.sort_values(
    ["region", "target_time"]
).reset_index(drop=True)

# Remove rows containing missing values
df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

print("Rows after cleaning:", len(df))


# ============================================================
# 5. CREATE LSTM SEQUENCES
# ============================================================

print("\nCreating LSTM sequences...")

variables = ["VHM0", "VTPK", "VPED"]
lags = list(range(8, 0, -1))

feature_columns = []

for lag in lags:
    for variable in variables:
        feature_columns.append(f"{variable}_t-{lag}")

print("\nLSTM input columns:")
print(feature_columns)

# Extract features
X_raw = df[feature_columns].values.astype(np.float32)

# Target
y = df["target_VHM0"].values.astype(np.float32)

# Reshape:
# samples x 24 features
# into
# samples x 8 timesteps x 3 variables
X = X_raw.reshape(
    len(df),
    8,
    3
)

print("\nLSTM input shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 6. CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

print("\nCreating chronological train/test split...")

# 80% training
# 20% testing

split_index = int(len(X) * 0.80)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. SCALE FEATURES
# ============================================================

print("\nScaling input features...")

scaler = StandardScaler()

# Fit scaler ONLY on training data
X_train_2d = X_train.reshape(
    -1,
    X_train.shape[-1]
)

X_test_2d = X_test.reshape(
    -1,
    X_test.shape[-1]
)

scaler.fit(X_train_2d)

X_train_scaled = scaler.transform(
    X_train_2d
).reshape(X_train.shape)

X_test_scaled = scaler.transform(
    X_test_2d
).reshape(X_test.shape)


# ============================================================
# 8. BUILD LSTM MODEL
# ============================================================

print("\nBuilding LSTM model...")

model = Sequential([
    LSTM(
        64,
        input_shape=(8, 3),
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ============================================================
# 9. TRAIN
# ============================================================

print("\nStarting LSTM training...")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# 10. PREDICTION
# ============================================================

print("\nTesting LSTM...")

y_pred = model.predict(
    X_test_scaled,
    verbose=0
).flatten()


# ============================================================
# 11. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

# Avoid division by zero
nonzero = np.abs(y_test) > 1e-8

mape = np.mean(
    np.abs(
        (y_test[nonzero] - y_pred[nonzero])
        / y_test[nonzero]
    )
) * 100


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LSTM RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")
print(f"MAPE : {mape:.2f}%")

print("=" * 60)


# ============================================================
# 13. SAVE MODEL
# ============================================================

model.save(MODEL_FILE)

joblib.dump(
    scaler,
    SCALER_FILE
)

print("\nSaved:")
print("1.", MODEL_FILE)
print("2.", SCALER_FILE)


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

test_df = df.iloc[split_index:].copy()

prediction_results = pd.DataFrame({
    "region": test_df["region"].values,
    "target_time": test_df["target_time"].values,
    "actual_VHM0": y_test,
    "predicted_VHM0": y_pred,
    "absolute_error": np.abs(y_test - y_pred)
})

prediction_results.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print("3.", PREDICTIONS_FILE)


# ============================================================
# 15. SAVE METRICS
# ============================================================

results = pd.DataFrame({
    "Model": ["LSTM"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2],
    "MAPE_percent": [mape]
})

results.to_csv(
    RESULTS_FILE,
    index=False
)

print("4.", RESULTS_FILE)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("LSTM TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)