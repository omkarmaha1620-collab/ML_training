import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

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

DATA_FILE = "xgboost_randomforest_dataset.csv"

MODEL_FILE = "random_forest_vessel_risk_v2.pkl"

RESULTS_FILE = "random_forest_vessel_risk_v2_results.csv"

SEED = 42


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 65)
print("RANDOM FOREST VESSEL RISK V2")
print("=" * 65)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES
# ============================================================

# These are the useful features identified from the
# previous dataset inspection.

feature_columns = [
    "position.lat",
    "position.lon",
    "fishing.totalDistanceKm",
    "fishing.averageSpeedKnots",
    "distances.startDistanceFromShoreKm",
    "distances.endDistanceFromShoreKm",
    "distances.startDistanceFromPortKm",
    "distances.endDistanceFromPortKm"
]


target_column = "target_risk"


# ============================================================
# 3. CHECK COLUMNS
# ============================================================

missing = [
    column
    for column in feature_columns + [target_column]
    if column not in df.columns
]

if missing:

    print("\nERROR: Missing columns:")

    for column in missing:
        print(column)

    raise ValueError(
        "Required columns are missing."
    )


# ============================================================
# 4. CLEAN DATA
# ============================================================

print("\nCleaning dataset...")

df = df[
    feature_columns + [target_column]
].copy()

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

before = len(df)

df = df.dropna().reset_index(drop=True)

after = len(df)

print(
    "Rows before cleaning:",
    before
)

print(
    "Rows after cleaning:",
    after
)


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 65)
print("TARGET DISTRIBUTION")
print("=" * 65)

print(
    df[target_column].value_counts()
)

print("\nPercent:")

print(
    (
        df[target_column]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


if df[target_column].nunique() < 2:

    raise ValueError(
        "Target contains only one class."
    )


# ============================================================
# 6. FEATURES / TARGET
# ============================================================

X = df[
    feature_columns
].astype(
    np.float32
)

y = df[
    target_column
].astype(
    int
)


print("\nFeatures used:")

for i, feature in enumerate(
    feature_columns,
    start=1
):
    print(
        f"{i}. {feature}"
    )


# ============================================================
# 7. STRATIFIED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 65)
print("STRATIFIED TRAIN / TEST SPLIT")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=SEED,

    stratify=y
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


print("\nTraining target:")

print(
    y_train.value_counts()
)


print("\nTesting target:")

print(
    y_test.value_counts()
)


# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 65)
print("TRAINING RANDOM FOREST...")
print("=" * 65)

model = RandomForestClassifier(

    n_estimators=500,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    class_weight="balanced",

    random_state=SEED,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Random Forest training completed."
)


# ============================================================
# 9. PREDICTION
# ============================================================

print("\nTesting model...")

y_probability = model.predict_proba(
    X_test
)[:, 1]

y_prediction = model.predict(
    X_test
)


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

print("\n" + "=" * 65)
print("RANDOM FOREST V2 RESULTS")
print("=" * 65)

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


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_prediction,
        labels=[0, 1],
        target_names=[
            "Normal",
            "Risk"
        ],
        zero_division=0
    )
)


print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_prediction,
    labels=[0, 1]
)

print(cm)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE")
print("=" * 65)

importance_df = pd.DataFrame({

    "feature":
        feature_columns,

    "importance":
        model.feature_importances_

})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved:")

print(
    MODEL_FILE
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Random Forest Vessel Risk V2"
    ],

    "Features": [
        len(feature_columns)
    ],

    "Training_Samples": [
        len(X_train)
    ],

    "Testing_Samples": [
        len(X_test)
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


print("\nResults saved:")

print(
    RESULTS_FILE
)


# ============================================================
# 15. SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = (
    "random_forest_vessel_risk_v2_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


print(
    "Feature importance saved:"
)

print(
    importance_file
)


# ============================================================
# 16. FINAL
# ============================================================

print("\n" + "=" * 65)
print("RANDOM FOREST VESSEL RISK V2 COMPLETED")
print("=" * 65)