import pandas as pd
import joblib


MODEL_PATH = "models/best_xgb_model.pkl"
SCALER_PATH = "models/scaler.pkl"


print("\n" + "-" * 60)
print("ŞİRKET İFLAS TAHMİN MODELİ")
print("-" * 60)


# Modeli yükle
model = joblib.load(MODEL_PATH)

# Scaler'ı yükle
scaler = joblib.load(SCALER_PATH)

print("Model başarıyla yüklendi.")
print("Scaler başarıyla yüklendi.")