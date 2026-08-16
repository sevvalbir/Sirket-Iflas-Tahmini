''' veri yükleme
sütun isimlerini temizleme
duplicate kontrolü
gereksiz değişkenleri kaldırma 
train/test ayrımı
preprocessing
scaling işlemleri
'''

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/data.csv"

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print('-'*60)
print("Veri Seti İle İlgili Bilgiler")
print('-'*60)

print(f"Satır sayısı : {df.shape[0]}")
print(f"Sütun sayısı : {df.shape[1]}")

print("\nİlk 5 kayıt:")
print(df.head())

print("\nEksik değer sayısı:")
print(df.isnull().sum().sum())

print("\nDuplicate kayıt sayısı:")
print(df.duplicated().sum())

print("\nHedef değişken dağılımı:")
print(df["Bankrupt?"].value_counts())


#features ve target belirliyoruz
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

#ayırdığımız sonuçları kontrol ediyoruz 
print("\n" + "-" * 60)
print("TRAIN / TEST SPLIT")
print("-" * 60)

print(f"X_train boyutu : {X_train.shape}")
print(f"X_test boyutu  : {X_test.shape}")

print(f"y_train boyutu : {y_train.shape}")
print(f"y_test boyutu  : {y_test.shape}")

print("\n" + "-" * 60)
print("DEĞİŞKEN TİPLERİ")
print("-" * 60)

print(X_train.dtypes.value_counts())

print("\nSayısal olmayan değişkenler:")

non_numeric_columns = X_train.select_dtypes(
    exclude=["number"]
).columns

print(non_numeric_columns.tolist())

print(f"\nSayısal olmayan değişken sayısı: {len(non_numeric_columns)}")

print("\nTrain hedef dağılımı:")
print(y_train.value_counts())

print("\nTest hedef dağılımı:")
print(y_test.value_counts())

print("\nTrain hedef oranları:")
print(y_train.value_counts(normalize=True))

print("\nTest hedef oranları:")
print(y_test.value_counts(normalize=True))



scaler = StandardScaler()

#scaler train üzerinden öğreniliyor
X_train_scaled = scaler.fit_transform(X_train)

#test verisi scaler'a öğretmeyeceğiz, train verisi üzerinden dönüştürülecek
X_test_scaled = scaler.transform(X_test)


print("\n" + "-" * 60)
print("SCALING")
print("-" * 60)

print(f"X_train_scaled boyutu : {X_train_scaled.shape}")
print(f"X_test_scaled boyutu  : {X_test_scaled.shape}")

print("\nTrain scaled ortalamalarının ilk 10'u:")
print(X_train_scaled.mean(axis=0)[:10])

print("\nTrain scaled standart sapmalarının ilk 10'u:")
print(X_train_scaled.std(axis=0)[:10])