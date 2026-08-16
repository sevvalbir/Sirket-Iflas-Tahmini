import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/raw/data.csv"

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

scaler.fit(X_train)


joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler başarıyla kaydedildi.")