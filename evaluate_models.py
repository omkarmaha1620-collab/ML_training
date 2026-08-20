import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ==============================
# 1. LOAD TEST DATA
# ==============================

df = pd.read_csv("test_dataset.csv")

X_test = df.drop(columns=["target_risk"])
y_test = df["target_risk"]


# ==============================
# 2. LOAD TRAINED MODELS
# ==============================

rf = joblib.load("random_forest_model.pkl")
xgb = joblib.load("xgboost_model.pkl")


# ==============================
# 3. FUNCTION FOR EVALUATION
# ==============================

def evaluate_model(name, model):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4) if auc is not None else "N/A")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    }


# ==============================
# 4. EVALUATE BOTH MODELS
# ==============================

rf_result = evaluate_model("Random Forest", rf)

xgb_result = evaluate_model("XGBoost", xgb)


# ==============================
# 5. SAVE RESULTS
# ==============================

results = pd.DataFrame([
    rf_result,
    xgb_result
])

results.to_csv("evaluation_results.csv", index=False)


# ==============================
# 6. FINAL COMPARISON
# ==============================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results.to_string(index=False))

print("\nEvaluation completed successfully!")
print("Saved: evaluation_results.csv")