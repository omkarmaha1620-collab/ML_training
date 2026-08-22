import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = r"C:\ML_training\44091.txt"

MODEL_FILE = r"C:\ML_training\LSTM\lstm_model_30min.keras"
SCALER_FILE = r"C:\ML_training\LSTM\lstm_scaler_30min.pkl"
RESULTS_FILE = r"C:\ML_training\LSTM\lstm_results_30min.csv"
PREDICTIONS_FILE = r"C:\ML_training\LSTM\lstm_predictions_30min.csv"

SEED = 42

# 8 observations × 30 minutes = 4 hours
SEQUENCE_LENGTH = 8

# NDBC expected interval
INTERVAL_MINUTES = 30

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    r"C:\ML_training\LSTM",
    exist_ok=True
)


# ============================================================
# 1. LOAD NDBC DATA
# ============================================================

print("=" * 70)
print("30-MINUTE NDBC LSTM TRAINING")
print("=" * 70)

print("\nLoading:", DATA_FILE)

df = pd.read_csv(
    DATA_FILE,
    sep=r"\s+",
    comment="#",
    header=None
)

print("Raw shape:", df.shape)


# ============================================================
# 2. NDBC COLUMN NAMES
# ============================================================

columns = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "WDIR",
    "WSPD",
    "GST",
    "WVHT",
    "DPD",
    "APD",
    "MWD",
    "PRES",
    "ATMP",
    "WTMP",
    "DEWP",
    "VIS",
    "PTDY",
    "TIDE"
]

df.columns = columns


# ============================================================
# 3. CREATE TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    {
        "year": df["year"],
        "month": df["month"],
        "day": df["day"],
        "hour": df["hour"],
        "minute": df["minute"]
    },
    errors="coerce"
)


# ============================================================
# 4. SELECT LSTM VARIABLES
# ============================================================

# NDBC → LSTM
#
# WVHT = significant wave height
# DPD  = dominant wave period
# MWD  = mean wave direction
#
# They are mapped to:
#
# WVHT → VHM0
# DPD  → VTPK
# MWD  → VPED

df = df[
    [
        "timestamp",
        "WVHT",
        "DPD",
        "MWD"
    ]
].copy()


# ============================================================
# 5. CONVERT VALUES TO NUMERIC
# ============================================================

for column in [
    "WVHT",
    "DPD",
    "MWD"
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 6. REMOVE INVALID OBSERVATIONS
# ============================================================

df = df.dropna(
    subset=[
        "timestamp",
        "WVHT",
        "DPD",
        "MWD"
    ]
).copy()


# ============================================================
# 7. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


print("\nValid observations:", len(df))

print(
    "Date range:",
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)


# ============================================================
# 8. CHECK TIME INTERVALS
# ============================================================

df["time_difference"] = (
    df["timestamp"]
    .diff()
)

print("\nTime interval distribution:")

print(
    df["time_difference"]
    .value_counts()
    .head(10)
)


# ============================================================
# 9. REMOVE DUPLICATE TIMESTAMPS
# ============================================================

df = df.drop_duplicates(
    subset=["timestamp"],
    keep="first"
).reset_index(
    drop=True
)


# ============================================================
# 10. CREATE COMPLETE 30-MINUTE GRID
# ============================================================

print("\nChecking 30-minute continuity...")

full_time_index = pd.date_range(
    start=df["timestamp"].min(),
    end=df["timestamp"].max(),
    freq="30min"
)

df = (
    df.set_index("timestamp")
    .reindex(full_time_index)
)

df.index.name = "timestamp"

df = df.reset_index()


# ============================================================
# 11. INTERPOLATE SMALL GAPS
# ============================================================

print(
    "\nMissing values before interpolation:"
)

print(
    df[
        ["WVHT", "DPD", "MWD"]
    ]
    .isna()
    .sum()
)


# Only interpolate short gaps.
#
# limit=2 means maximum two consecutive
# missing 30-minute observations.

df[
    ["WVHT", "DPD", "MWD"]
] = (
    df[
        ["WVHT", "DPD", "MWD"]
    ]
    .interpolate(
        method="linear",
        limit=2
    )
)


# ============================================================
# 12. REMOVE REMAINING MISSING VALUES
# ============================================================

df = df.dropna(
    subset=[
        "WVHT",
        "DPD",
        "MWD"
    ]
).reset_index(
    drop=True
)


print(
    "\nValid continuous observations:",
    len(df)
)


# ============================================================
# 13. CREATE LSTM SEQUENCES
# ============================================================

print("\nCreating 30-minute LSTM sequences...")

features = [
    "WVHT",
    "DPD",
    "MWD"
]

X_sequences = []
y_targets = []
timestamps = []


for i in range(
    SEQUENCE_LENGTH,
    len(df)
):

    history = df[
        features
    ].iloc[
        i - SEQUENCE_LENGTH:i
    ].values

    target = df[
        "WVHT"
    ].iloc[
        i
    ]

    target_time = df[
        "timestamp"
    ].iloc[
        i
    ]

    X_sequences.append(
        history
    )

    y_targets.append(
        target
    )

    timestamps.append(
        target_time
    )


X = np.array(
    X_sequences,
    dtype=np.float32
)

y = np.array(
    y_targets,
    dtype=np.float32
)

timestamps = np.array(
    timestamps
)


print("\nX shape:", X.shape)

print(
    "Expected:",
    "(samples, 8, 3)"
)

print(
    "y shape:",
    y.shape
)


# ============================================================
# 14. CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

split_index = int(
    len(X) * 0.80
)

X_train = X[
    :split_index
]

X_test = X[
    split_index:
]

y_train = y[
    :split_index
]

y_test = y[
    split_index:
]

time_test = timestamps[
    split_index:
]


print("\n" + "=" * 70)
print("TRAIN / TEST")
print("=" * 70)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 15. SCALE INPUT FEATURES
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_2d = X_train.reshape(
    -1,
    3
)

X_test_2d = X_test.reshape(
    -1,
    3
)

scaler.fit(
    X_train_2d
)

X_train_scaled = (
    scaler
    .transform(X_train_2d)
    .reshape(X_train.shape)
)

X_test_scaled = (
    scaler
    .transform(X_test_2d)
    .reshape(X_test.shape)
)


# ============================================================
# 16. BUILD LSTM
# ============================================================

print("\nBuilding LSTM...")

model = Sequential([

    LSTM(
        64,
        input_shape=(
            SEQUENCE_LENGTH,
            3
        ),
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1
    )
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


model.summary()


# ============================================================
# 17. TRAIN
# ============================================================

print("\nTraining...")

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

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# 18. PREDICTION
# ============================================================

print("\nTesting LSTM...")

y_pred = model.predict(
    X_test_scaled,
    verbose=0
).flatten()


# ============================================================
# 19. EVALUATION
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


nonzero = (
    np.abs(y_test)
    > 1e-8
)

mape = np.mean(
    np.abs(
        (
            y_test[nonzero]
            -
            y_pred[nonzero]
        )
        /
        y_test[nonzero]
    )
) * 100


# ============================================================
# 20. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("30-MINUTE LSTM RESULTS")
print("=" * 70)

print(
    f"MAE  : {mae:.6f} m"
)

print(
    f"RMSE : {rmse:.6f} m"
)

print(
    f"R²   : {r2:.6f}"
)

print(
    f"MAPE : {mape:.2f}%"
)


# ============================================================
# 21. SAVE MODEL
# ============================================================

model.save(
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved scaler:")
print(SCALER_FILE)


# ============================================================
# 22. SAVE PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({

    "timestamp":
        time_test,

    "actual_VHM0":
        y_test,

    "predicted_VHM0":
        y_pred,

    "absolute_error":
        np.abs(
            y_test - y_pred
        )
})


prediction_results.to_csv(
    PREDICTIONS_FILE,
    index=False
)


print(
    "\nPredictions saved:"
)

print(
    PREDICTIONS_FILE
)


# ============================================================
# 23. SAVE METRICS
# ============================================================

results = pd.DataFrame({

    "Model":
        [
            "LSTM_30min_NDBC"
        ],

    "Interval_minutes":
        [
            30
        ],

    "Sequence_length":
        [
            SEQUENCE_LENGTH
        ],

    "History_hours":
        [
            SEQUENCE_LENGTH
            * INTERVAL_MINUTES
            / 60
        ],

    "MAE":
        [
            mae
        ],

    "RMSE":
        [
            rmse
        ],

    "R2":
        [
            r2
        ],

    "MAPE_percent":
        [
            mape
        ]
})


results.to_csv(
    RESULTS_FILE,
    index=False
)


print(
    "\nResults saved:"
)

print(
    RESULTS_FILE
)


# ============================================================
# 24. FINAL
# ============================================================

print("\n" + "=" * 70)
print("30-MINUTE LSTM TRAINING COMPLETED")
print("=" * 70)

print(
    "\nModel meaning:"
)

print(
    "8 × 30-minute observations"
)

print(
    "→ 4 hours of recent wave history"
)

print(
    "→ predicts the next 30-minute wave height"
)

print(
    "\nNDBC mapping:"
)

print(
    "WVHT → VHM0"
)

print(
    "DPD  → VTPK"
)

print(
    "MWD  → VPED"
)