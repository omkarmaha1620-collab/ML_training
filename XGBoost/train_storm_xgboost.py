import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "XGBoost/data/ndbc_high_wave_dataset.csv"

MODEL_FILE = "XGBoost/xgboost_high_wave_ndbc_model.pkl"

RESULTS_FILE = "XGBoost/xgboost_high_wave_ndbc_results.csv"

HIGH_WAVE_THRESHOLD = 3.0

SEED = 42


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("XGBOOST HIGH-WAVE HAZARD MODEL")
print("NDBC DATASET")
print("=" * 60)

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["timestamp"]
)

print("\nDataset shape:", df.shape)

print(
    "Date range:",
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)


# ============================================================
# 2. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# 3. FEATURE COLUMNS
# ============================================================

base_features = [
    "WVHT",
    "WSPD",
    "GST",
    "DPD",
    "APD",
    "PRES",
    "ATMP",
    "WTMP"
]

feature_columns = []

for lag in [1, 2, 3]:

    for feature in base_features:

        feature_columns.append(
            f"{feature}_t-{lag}"
        )


missing = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing:

    print("\nMissing features:")

    for column in missing:
        print(column)

    raise ValueError(
        "Required features are missing."
    )


# ============================================================
# 4. TARGET
# ============================================================

X = df[
    feature_columns
].astype(
    np.float32
)

y = df[
    "high_wave"
].astype(
    int
)


print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(
    y.value_counts()
)

print(
    "\nPercent:"
)

print(
    (
        y.value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ============================================================
# 5. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# First 80% of time → training
# Last 20% of time  → testing

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print("\n" + "=" * 60)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 60)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "\nTraining period:"
)

print(
    df["timestamp"].iloc[0],
    "to",
    df["timestamp"].iloc[
        split_index - 1
    ]
)

print(
    "\nTesting period:"
)

print(
    df["timestamp"].iloc[
        split_index
    ],
    "to",
    df["timestamp"].iloc[-1]
)


# ============================================================
# 6. CHECK BOTH CLASSES
# ============================================================

print(
    "\nTraining target:"
)

print(
    y_train.value_counts()
)

print(
    "\nTesting target:"
)

print(
    y_test.value_counts()
)

if y_train.nunique() < 2:

    raise ValueError(
        "Training set contains only one class."
    )

if y_test.nunique() < 2:

    raise ValueError(
        "Testing set contains only one class. "
        "A meaningful ROC-AUC cannot be calculated."
    )


# ============================================================
# 7. CLASS IMBALANCE
# ============================================================

negative = (
    y_train == 0
).sum()

positive = (
    y_train == 1
).sum()

scale_pos_weight = (
    negative / positive
)

print(
    "\nScale positive weight:",
    scale_pos_weight
)


# ============================================================
# 8. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST...")
print("=" * 60)

model = XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=SEED,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# 9. PREDICTION
# ============================================================

print(
    "\nTesting model..."
)

y_probability = model.predict_proba(
    X_test
)[:, 1]

y_prediction = (
    y_probability >= 0.50
).astype(int)


# ============================================================
# 10. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 11. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST NDBC RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_prediction,
        zero_division=0
    )
)


print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_prediction,
        labels=[0, 1]
    )
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print(
    "\nModel saved:"
)

print(
    MODEL_FILE
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "Model": [
        "XGBoost NDBC High-Wave Detector"
    ],

    "Threshold_VHM0_m": [
        HIGH_WAVE_THRESHOLD
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1": [
        f1
    ],

    "ROC_AUC": [
        roc_auc
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
# 14. FINAL
# ============================================================

print("\n" + "=" * 60)
print("NDBC XGBOOST TRAINING COMPLETED")
print("=" * 60)