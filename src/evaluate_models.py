import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


DATA_PATH = "data/raw/data.csv"
MODEL_PATH = "models/best_xgb_model.pkl"


df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\n" + "-" * 60)
print("KAYDEDİLMİŞ XGBOOST MODELİ")
print("-" * 60)

model = joblib.load(MODEL_PATH)

print("Model başarıyla yüklendi.")



y_test_proba = model.predict_proba(
    X_test_scaled
)[:, 1]


final_threshold = 0.40

y_test_pred = (
    y_test_proba >= final_threshold
).astype(int)


# ============================================================
# 1.MODEL PERFORMANSI
# ============================================================

print("\n" + "-" * 60)
print("FINAL MODEL PERFORMANSI")
print("-" * 60)

cm = confusion_matrix(
    y_test,
    y_test_pred
)

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_test_proba
)

pr_auc = average_precision_score(
    y_test,
    y_test_proba
)


print("\nConfusion Matrix:")
print(cm)

print("\nModel Performansı:")
print(f"Threshold : {final_threshold:.2f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)


# ============================================================
# 2.CONFUSION MATRIX GRAFİĞİ
# ============================================================

print("\n" + "-" * 60)
print("CONFUSION MATRIX GRAFİĞİ")
print("-" * 60)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["İflas Etmedi", "İflas Etti"]
)

disp.plot()

plt.title("Final XGBoost - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 3.ROC CURVE
# ============================================================

print("\n" + "-" * 60)
print("ROC CURVE")
print("-" * 60)

fpr, tpr, thresholds_roc = roc_curve(
    y_test,
    y_test_proba
)

roc_auc_value = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc_value:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Final XGBoost")

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 4.PRECISION - RECALL CURVE
# ============================================================

print("\n" + "-" * 60)
print("PRECISION-RECALL CURVE")
print("-" * 60)

precision_curve, recall_curve, thresholds_pr = (
    precision_recall_curve(
        y_test,
        y_test_proba
    )
)

pr_auc_value = average_precision_score(
    y_test,
    y_test_proba
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"XGBoost (PR-AUC = {pr_auc_value:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Final XGBoost"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 5.FEATURE IMPORTANCE
# ============================================================

print("\n" + "-" * 60)
print("FEATURE IMPORTANCE")
print("-" * 60)


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nEn önemli 20 değişken:")

print(
    feature_importance.head(20).to_string(
        index=False
    )
)


# ------------------------------------------------------------
#FEATURE IMPORTANCE GRAFİĞİ
# ------------------------------------------------------------

top_features = feature_importance.head(20)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Final XGBoost - En Önemli 20 Değişken"
)

plt.tight_layout()

plt.show()


print("\n" + "-" * 60)
print("DEĞERLENDİRME TAMAMLANDI")
print("-" * 60)