import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_FILE = "xgboost_randomforest_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. PREPARE FEATURES AND TARGET
# ==========================================

TARGET = "target_risk"

# Keep only numeric columns
numeric_df = df.select_dtypes(include="number").copy()

if TARGET not in numeric_df.columns:
    raise ValueError("target_risk column was not found!")

X = numeric_df.drop(columns=[TARGET])
y = numeric_df[TARGET].astype(int)

print("\n" + "=" * 60)
print("FEATURES")
print("=" * 60)
print("Number of features:", X.shape[1])
print("Features:", X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. HANDLE CLASS IMBALANCE
# ==========================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nClass imbalance ratio:", scale_pos_weight)


# ==========================================
# 5. RANDOM FOREST
# ==========================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST...")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", rf_accuracy)
print("ROC-AUC:", rf_auc)

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# Save Random Forest
joblib.dump(rf_model, "random_forest_model.pkl")


# ==========================================
# 6. XGBOOST
# ==========================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST...")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_prob)

print("\nXGBOOST RESULTS")
print("Accuracy:", xgb_accuracy)
print("ROC-AUC:", xgb_auc)

print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))


# Save XGBoost
joblib.dump(xgb_model, "xgboost_model.pkl")


# ==========================================
# 7. SAVE TEST DATA
# ==========================================

test_data = X_test.copy()
test_data[TARGET] = y_test

test_data.to_csv(
    "test_dataset.csv",
    index=False
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [rf_accuracy, xgb_accuracy],
    "ROC_AUC": [rf_auc, xgb_auc]
})

results.to_csv(
    "model_results.csv",
    index=False
)


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nModels saved:")
print("1. random_forest_model.pkl")
print("2. xgboost_model.pkl")

print("\nTesting dataset:")
print("3. test_dataset.csv")

print("\nResults:")
print("4. model_results.csv")

print("\n" + results.to_string(index=False))

print("\nDONE!")