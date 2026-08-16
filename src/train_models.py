#ilk önce logistic regression modelini deneyeceğiz

import pandas as pd 
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold


DATA_PATH = "data/raw/data.csv"

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

#train ve test setlerini ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#scaling yapacağız
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#-------------------------------------
#LOGISTIC REGRESSION MODELİ İÇİN: 
#-------------------------------------


model = LogisticRegression(
    class_weight="balanced",  #azınlık olan iflas sınıfını dengelemek için. 
    max_iter=1000,
    random_state=42
)

#modeli eğitiyoruz 
model.fit(X_train_scaled, y_train)

#tahminleme yapıyoruz 
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

#sonuçları inceliyoruz
print("\n" + "-" * 60)
print("LOGISTIC REGRESSION")
print("-" * 60)

print(f"Model eğitildi.")
print(f"Train gözlem sayısı : {X_train_scaled.shape[0]}")
print(f"Test gözlem sayısı  : {X_test_scaled.shape[0]}")

print("\nİlk 20 tahmin:")
print(y_pred[:20])

print("\nİlk 20 iflas olasılığı:")
print(y_pred_proba[:20])

#buradan sonra çıktıya bakıldığında veri seti dengesiz olduğundan başarılı bir sonuç gelmedi.



#modeli değerlendirirsek: 
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


#Confusion Matrix oluşturalım
cm = confusion_matrix(y_test, y_pred)

print("\n" + "-" * 60)
print("CONFUSION MATRIX")
print("-" * 60)

print(cm)


#temel metriklere bakacağız
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#ROC-AUC bakalım
roc_auc = roc_auc_score(y_test, y_pred_proba)

#PR-AUC bakalım
pr_auc = average_precision_score(y_test, y_pred_proba)


print("\n" + "-" * 60)
print("MODEL PERFORMANSI")
print("-" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


#Sonuç olarak reportu inceleyelim
print("\n" + "-" * 60)
print("CLASSIFICATION REPORT")
print("-" * 60)

print(classification_report(y_test, y_pred))




#-------------------------------------
#XGBOOST MODELİ İÇİN: 
#-------------------------------------

model_xgb = XGBClassifier(
    n_estimators=300,  #300 model ağaç oluşturuyoruz 
    max_depth=4,  #overfitting riskini azaltmak için  
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1  #işlem için mevcut cpu kullanılıyoruz 
)

#modeli eğitiyoruz 
model_xgb.fit(X_train_scaled, y_train) 


#tahminleme yapıyoruz 
y_pred_xgb = model_xgb.predict(X_test_scaled)
y_pred_proba_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]

cm_xgb = confusion_matrix(y_test, y_pred_xgb)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

roc_auc_xgb = roc_auc_score(
    y_test,
    y_pred_proba_xgb
)

pr_auc_xgb = average_precision_score(
    y_test,
    y_pred_proba_xgb
)


print("\n" + "-" * 60)
print("XGBOOST")
print("-" * 60)

print("Model eğitildi.")

print("\nConfusion Matrix:")
print(cm_xgb)

print("\nModel Performansı:")

print(f"Accuracy  : {accuracy_xgb:.4f}")
print(f"Precision : {precision_xgb:.4f}")
print(f"Recall    : {recall_xgb:.4f}")
print(f"F1-Score  : {f1_xgb:.4f}")
print(f"ROC-AUC   : {roc_auc_xgb:.4f}")
print(f"PR-AUC    : {pr_auc_xgb:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_xgb
    )
)




#-------------------------------------
#XGBOOST MODELİ + CLASS IMBALANCE İÇİN:
#-------------------------------------

scale_pos_weight = (
    y_train.value_counts()[0] /
    y_train.value_counts()[1]
)

print("\n" + "-" * 60)
print("XGBOOST + CLASS IMBALANCE")
print("-" * 60)

print(f"Scale Pos Weight : {scale_pos_weight:.2f}")


model_xgb_balanced = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

#modeli eğitiyoruz 
model_xgb_balanced.fit(
    X_train_scaled,
    y_train
)

#tahminleme yapıyoruz
y_pred_xgb_balanced = model_xgb_balanced.predict(
    X_test_scaled
)

y_pred_proba_xgb_balanced = (
    model_xgb_balanced.predict_proba(X_test_scaled)[:, 1]
)


cm_xgb_balanced = confusion_matrix(
    y_test,
    y_pred_xgb_balanced
)

accuracy_xgb_balanced = accuracy_score(
    y_test,
    y_pred_xgb_balanced
)

precision_xgb_balanced = precision_score(
    y_test,
    y_pred_xgb_balanced
)

recall_xgb_balanced = recall_score(
    y_test,
    y_pred_xgb_balanced
)

f1_xgb_balanced = f1_score(
    y_test,
    y_pred_xgb_balanced
)

roc_auc_xgb_balanced = roc_auc_score(
    y_test,
    y_pred_proba_xgb_balanced
)

pr_auc_xgb_balanced = average_precision_score(
    y_test,
    y_pred_proba_xgb_balanced
)

print("\nConfusion Matrix:")
print(cm_xgb_balanced)

print("\nModel Performansı:")

print(f"Accuracy  : {accuracy_xgb_balanced:.4f}")
print(f"Precision : {precision_xgb_balanced:.4f}")
print(f"Recall    : {recall_xgb_balanced:.4f}")
print(f"F1-Score  : {f1_xgb_balanced:.4f}")
print(f"ROC-AUC   : {roc_auc_xgb_balanced:.4f}")
print(f"PR-AUC    : {pr_auc_xgb_balanced:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_xgb_balanced
    )
)


#-------------------------------------
#XGBOOST + CLASS IMBALANCE HYPERPARAMETER OPTIMIZATION
#-------------------------------------

print("\n" + "-" * 60)
print("XGBOOST HYPERPARAMETER OPTIMIZATION")
print("-" * 60)


# Sınıf dengesizliği ağırlığı
scale_pos_weight = (
    y_train.value_counts()[0] /
    y_train.value_counts()[1]
)

print(f"Scale Pos Weight : {scale_pos_weight:.2f}")


#Temel XGBOOST modeli
xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

#HYPERPARAMETER ARAMA ALANI 
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5]
}

#STRATIFIED CROSS VALIDATION
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

#RANDOMIZED SEARCH
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=30,
    scoring="f1",
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

#modeli eğitiyoruz 
random_search.fit(
    X_train_scaled,
    y_train
)


#en iyi parametreleri ve en iyi modeli yazdırıyoruz 
print("\n" + "-" * 60)
print("EN İYİ PARAMETRELER")
print("-" * 60)

print(random_search.best_params_)

print(
    f"\nCross Validation F1-Score : "
    f"{random_search.best_score_:.4f}"
)


best_xgb_model = random_search.best_estimator_


print("\nEn iyi model:")
print(best_xgb_model)

# En iyi modeli kaydedeceğiz ki kod her seferinde tekrar çalışmasın
joblib.dump(
    best_xgb_model,
    "models/best_xgb_model.pkl"
)

print("\nEn iyi XGBoost modeli kaydedildi:")
print("models/best_xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler başarıyla kaydedildi.")

#-------------------------------------
#RANDOM FOREST + CLASS IMBALANCE MODELİ İÇİN:
#-------------------------------------

model_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced", #xgboosttaki scale_pos_weightin karşılığı
    random_state=42,
    n_jobs=-1
)

#modeli eğitiyoruz
model_rf.fit(
    X_train_scaled,
    y_train
)

#tahminleme yapıyoruz
y_pred_rf = model_rf.predict(X_test_scaled)

y_pred_proba_rf = model_rf.predict_proba(
    X_test_scaled
)[:, 1]

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)

precision_rf = precision_score(
    y_test,
    y_pred_rf
)

recall_rf = recall_score(
    y_test,
    y_pred_rf
)

f1_rf = f1_score(
    y_test,
    y_pred_rf
)

roc_auc_rf = roc_auc_score(
    y_test,
    y_pred_proba_rf
)

pr_auc_rf = average_precision_score(
    y_test,
    y_pred_proba_rf
)

print("\n" + "-" * 60)
print("RANDOM FOREST + CLASS IMBALANCE")
print("-" * 60)

print("Model eğitildi.")

print("\nConfusion Matrix:")
print(cm_rf)

print("\nModel Performansı:")

print(f"Accuracy  : {accuracy_rf:.4f}")
print(f"Precision : {precision_rf:.4f}")
print(f"Recall    : {recall_rf:.4f}")
print(f"F1-Score  : {f1_rf:.4f}")
print(f"ROC-AUC   : {roc_auc_rf:.4f}")
print(f"PR-AUC    : {pr_auc_rf:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_rf
    )
)




#XGBOOST + CLASS IMBALANCE SEÇİYORUZ VE İYİLEŞTİRME YAPACAĞIZ
#-------------------------------------
#THRESHOLD OPTIMIZATION 
#-------------------------------------

print("\n" + "-" * 60)
print("THRESHOLD ANALİZİ")
print("-" * 60)

thresholds = np.arange(0.10, 0.71, 0.05)

threshold_results = []

for threshold in thresholds:

    y_pred_threshold = (
        y_pred_proba_xgb_balanced >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


threshold_df = pd.DataFrame(threshold_results)


print("\nThreshold sonuçları:")
print(threshold_df.to_string(index=False))


# En yüksek F1 değerine sahip threshold
best_threshold_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

print("\n" + "-" * 60)
print("EN İYİ THRESHOLD")
print("-" * 60)

print(
    f"Threshold : {best_threshold_row['Threshold']:.2f}"
)

print(
    f"Precision : {best_threshold_row['Precision']:.4f}"
)

print(
    f"Recall    : {best_threshold_row['Recall']:.4f}"
)

print(
    f"F1-Score  : {best_threshold_row['F1']:.4f}"
)



#-------------------------------------
#OPTIMIZED XGBOOST - TEST SET EVALUATION
#-------------------------------------

print("\n" + "-" * 60)
print("OPTIMIZED XGBOOST - TEST SET")
print("-" * 60)


# Test tahmini
y_pred_best_xgb = best_xgb_model.predict(
    X_test_scaled
)

y_pred_proba_best_xgb = (
    best_xgb_model.predict_proba(X_test_scaled)[:, 1]
)


cm_best_xgb = confusion_matrix(
    y_test,
    y_pred_best_xgb
)

accuracy_best_xgb = accuracy_score(
    y_test,
    y_pred_best_xgb
)

precision_best_xgb = precision_score(
    y_test,
    y_pred_best_xgb,
    zero_division=0
)

recall_best_xgb = recall_score(
    y_test,
    y_pred_best_xgb,
    zero_division=0
)

f1_best_xgb = f1_score(
    y_test,
    y_pred_best_xgb,
    zero_division=0
)

roc_auc_best_xgb = roc_auc_score(
    y_test,
    y_pred_proba_best_xgb
)

pr_auc_best_xgb = average_precision_score(
    y_test,
    y_pred_proba_best_xgb
)

print("\nConfusion Matrix:")
print(cm_best_xgb)

print("\nModel Performansı:")

print(f"Accuracy  : {accuracy_best_xgb:.4f}")
print(f"Precision : {precision_best_xgb:.4f}")
print(f"Recall    : {recall_best_xgb:.4f}")
print(f"F1-Score  : {f1_best_xgb:.4f}")
print(f"ROC-AUC   : {roc_auc_best_xgb:.4f}")
print(f"PR-AUC    : {pr_auc_best_xgb:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_best_xgb
    )
)


#-------------------------------------
#THRESHOLD OPTIMIZATION + CROSS VALIDATION 
#-------------------------------------

print("\n" + "-" * 60)
print("THRESHOLD OPTIMIZATION - CROSS VALIDATION")
print("-" * 60)


#Threshold değerleri
thresholds = np.arange(0.10, 0.71, 0.05)


#Stratified K-Fold
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

threshold_cv_results = []

for threshold in thresholds:

    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train_scaled, y_train),
        start=1
    ):

        # Fold train / validation
        X_fold_train = X_train_scaled[train_idx]
        X_fold_val = X_train_scaled[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

#Optimized XGBoost Modeli için: 
        cv_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,

            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.7,
            gamma=0,

            scale_pos_weight=scale_pos_weight
        )


        # Model eğitimi
        cv_model.fit(
            X_fold_train,
            y_fold_train
        )


        # İflas olasılıkları
        y_val_proba = cv_model.predict_proba(
            X_fold_val
        )[:, 1]


        # Threshold uygulama
        y_val_pred = (
            y_val_proba >= threshold
        ).astype(int)


        # Metrikler
        precision = precision_score(
            y_fold_val,
            y_val_pred,
            zero_division=0
        )

        recall = recall_score(
            y_fold_val,
            y_val_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_fold_val,
            y_val_pred,
            zero_division=0
        )


        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1_scores.append(f1)



    threshold_cv_results.append({
        "Threshold": round(threshold, 2),
        "Precision": np.mean(fold_precisions),
        "Recall": np.mean(fold_recalls),
        "F1": np.mean(fold_f1_scores)
    })


threshold_cv_df = pd.DataFrame(
    threshold_cv_results
)

print("\nThreshold CV sonuçları:")

print(
    threshold_cv_df.to_string(
        index=False
    )
)

best_threshold_row = threshold_cv_df.loc[
    threshold_cv_df["F1"].idxmax()
]


best_threshold = (
    best_threshold_row["Threshold"]
)


print("\n" + "-" * 60)
print("EN İYİ THRESHOLD - CROSS VALIDATION")
print("-" * 60)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Mean Precision : "
    f"{best_threshold_row['Precision']:.4f}"
)

print(
    f"Mean Recall    : "
    f"{best_threshold_row['Recall']:.4f}"
)

print(
    f"Mean F1-Score  : "
    f"{best_threshold_row['F1']:.4f}"
)


#-------------------------------------
#XGBOOST + GRIDSEARCH 
#-------------------------------------


print("\n" + "-" * 60)
print("XGBOOST GRIDSEARCHCV")
print("-" * 60)


xgb_grid_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [250, 300],
    "max_depth": [5, 6],
    "learning_rate": [0.02, 0.03],
    "min_child_weight": [4, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.7],
    "gamma": [0]
}

grid_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_grid_base,
    param_grid=param_grid,
    scoring="f1",
    cv=grid_cv,
    verbose=2,
    n_jobs=1
)


grid_search.fit(
    X_train_scaled,
    y_train
)


print("\n" + "-" * 60)
print("GRIDSEARCHCV EN İYİ PARAMETRELER")
print("-" * 60)

print(grid_search.best_params_)

print(
    f"\nCross Validation F1-Score : "
    f"{grid_search.best_score_:.4f}"
)


final_xgb_model = grid_search.best_estimator_

print("\nFinal aday XGBoost modeli:")
print(final_xgb_model)



# ---------------------------------------------------------
# FINAL GRIDSEARCH XGBOOST - TEST SET
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("FINAL GRIDSEARCH XGBOOST - TEST SET")
print("-" * 60)


# GridSearch tarafından bulunan en iyi model
final_xgb_model = grid_search.best_estimator_


final_xgb_model.fit(
    X_train_scaled,
    y_train
)


y_test_proba = final_xgb_model.predict_proba(
    X_test_scaled
)[:, 1]


final_threshold = 0.40 #Cross-validation ile belirlediğimiz threshold


# Threshold uygulama
y_test_pred = (
    y_test_proba >= final_threshold
).astype(int)



cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion Matrix:")
print(cm)



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


print("\nModel Performansı")

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



