# 📊 ŞİRKET İFLAS TAHMİNİ

> XGBoost tabanlı makine öğrenmesi modeli ile şirketlerin iflas riskinin tahmin edilmesi.

## 🚀 Live Dashboard

🔗[ **[Şirket İflas Tahmini Dashboard'u](https://sirket-iflas-tahmini-wus6sckfrvc2fmcbf4lvct.streamlit.app/)**](https://sirket-iflas-tahmini-wus6sckfrvc2fmcbf4lvct.streamlit.app/)

---

## 📌 Proje Hakkında

Bu proje, şirketlerin finansal göstergelerini kullanarak **iflas riskini tahmin etmek** amacıyla geliştirilmiştir.

Çalışmada şirketlerin finansal durumlarını temsil eden **95 farklı özellik** kullanılmış ve iflas tahmini için XGBoost tabanlı bir sınıflandırma modeli geliştirilmiştir.

Projenin son aşamasında geliştirilen **Streamlit dashboard** sayesinde veri setindeki şirket kayıtları seçilerek model üzerinden iflas olasılığı tahmin edilebilmektedir.

Bu proje kapsamında veri ön işleme, sınıflandırma, model optimizasyonu, sınıf dengesizliği, model değerlendirme ve modelin gerçek bir uygulama üzerinden kullanılabilir hale getirilmesi süreçleri ele alınmıştır.

---

## 🎯 Projenin Amacı

Şirketlerin finansal durumlarından yararlanarak:

- Şirketin iflas edip etmeyeceğini tahmin etmek
- İflas olasılığını hesaplamak
- İflas eden şirketleri mümkün olduğunca doğru tespit etmek
- Modelin performansını farklı sınıflandırma metrikleri ile değerlendirmek
- Eğitilmiş modeli interaktif bir dashboard üzerinden kullanılabilir hale getirmek amaçlanmıştır.

---

## 📂 Veri Seti

Projede **Company Bankruptcy Prediction** veri seti kullanılmıştır.

---

##📂 Veri Seti Özellikleri

- Toplam kayıt: 6.819
- Feature sayısı: 95
- Target: `Bankrupt?`
- İflas eden şirket: 220
- İflas etmeyen şirket: 6.599
Veri setindeki sınıfların belirgin şekilde dengesiz olması nedeniyle model geliştirme sürecinde **class imbalance** problemi ayrıca ele alınmıştır.

0 → İflas etmedi

1 → İflas etti

---

## 🔄 Veri Ön İşleme
Modelleme öncesinde veri seti üzerinde aşağıdaki işlemler gerçekleştirilmiştir:

Veri setinin incelenmesi

Sütun isimlerinin düzenlenmesi

Eksik değerlerin kontrol edilmesi

Duplicate kayıtların kontrol edilmesi

Bağımlı ve bağımsız değişkenlerin ayrılması

Eğitim ve test veri setlerinin oluşturulması

Feature'ların ölçeklendirilmesi

Sınıf dengesizliğinin incelenmesi

Veri setinde:

Eksik değer bulunmamaktadır.

Duplicate kayıt bulunmamaktadır.

Model eğitiminde toplam 95 bağımsız değişken kullanılmıştır.

---

## 🤖 Kullanılan Model
Projenin ana modeli:XGBoost
XGBoost, özellikle sınıflandırma problemlerinde güçlü performans sağlayan gradient boosting tabanlı bir ensemble learning algoritmasıdır.
Bu projede XGBoost tercih edilmesinin temel nedeni, finansal veriler gibi çok sayıda değişkene sahip veri setlerinde karmaşık ilişkileri öğrenebilmesi ve sınıflandırma performansının güçlü olmasıdır.
Model geliştirme sürecinde sınıf dengesizliği de dikkate alınmıştır.

---

## ⚖️ Class Imbalance
Veri setinde iflas eden şirketlerin sayısı:220
iflas etmeyen şirketlerin sayısı ise:6599

olduğu için ciddi bir sınıf dengesizliği bulunmaktadır.
Bu nedenle model değerlendirilirken yalnızca Accuracy metriğine odaklanılmamıştır.

Özellikle:

Precision

Recall

F1-Score

ROC-AUC

PR-AUC metrikleri dikkate alınmıştır.

İflas tahmini probleminde Recall özellikle önemlidir. Çünkü gerçekten iflas edecek bir şirketin model tarafından gözden kaçırılması önemli bir hatadır.

---

## 🔍 Model Optimizasyonu
XGBoost modeli için hiperparametre optimizasyonu gerçekleştirilmiştir.

Kullanılan optimizasyon sürecinde:
RandomizedSearchCV
5-fold Cross Validation
F1-Score kullanılmıştır.

En iyi parametreler:

subsample = 0.8

n_estimators = 300

min_child_weight = 5

max_depth = 6

learning_rate = 0.03

gamma = 0

colsample_bytree = 0.7

Bu optimizasyon sonucunda modelin özellikle azınlık sınıfındaki performansı iyileştirilmiştir.

---

## 📈 Final Model Performansı
Final XGBoost modeli test veri seti üzerinde değerlendirilmiştir.
Classification threshold: 0.40 olarak belirlenmiştir.

Test Sonuçları

Metric	Score

Accuracy	95.38%

Precision	37.66%

Recall	65.91%

F1-Score	47.93%

ROC-AUC	95.21%

PR-AUC	47.63%

Confusion Matrix:
[[1272   48]
 [  15   29]]

Bu sonuçlara göre model test setindeki 44 gerçek iflas vakasının 29'unu doğru şekilde tespit etmiştir.
Bu nedenle iflas sınıfındaki Recall değeri: 65.91% olarak elde edilmiştir.

---

## 🔎 Feature Importance
XGBoost modelinin en önemli değişkenleri incelendiğinde finansal yapı ile doğrudan ilişkili değişkenlerin öne çıktığı görülmüştür.
Model tarafından en önemli bulunan bazı feature'lar:

Feature	Importance

Continuous interest rate (after tax)	0.091695

Borrowing dependency	0.054851

Net Income to Total Assets	0.054115

Persistent EPS in the Last Four Seasons	0.052764

Per Share Net profit before tax (Yuan ¥)	0.039053

Net worth/Assets	0.031478

Debt ratio %	0.030024

Net Value Per Share (C)	0.024496

Retained Earnings to Total Assets	0.023767

Equity to Liability	0.021011

Bu sonuçlar modelin şirketlerin borçluluk, kârlılık, özkaynak ve faiz yükü gibi finansal yapılarını temsil eden değişkenlerden önemli ölçüde yararlandığını göstermektedir.

---

## 🖥️ Streamlit Dashboard

Modelin yalnızca Python kodu içerisinde çalışması yerine, kullanıcıların modeli interaktif şekilde kullanabilmesi amacıyla bir Streamlit dashboard geliştirilmiştir.

Dashboard üzerinden kullanıcı:

Veri setindeki bir şirket kaydını seçebilir.

Seçilen şirketin temel finansal göstergelerini görüntüleyebilir.

TAHMİN HESAPLA butonuyla model tahmini gerçekleştirebilir.

Tahmin edilen iflas olasılığını görebilir.

Seçilen şirketin 95 feature değerini inceleyebilir.


Dashboard'da kullanılan model:XGBoost ve sınıflandırma threshold değeri:0.40 olarak belirlenmiştir.

---

## 🗂️ Proje Yapısı
```text
Sirket-Iflas-Tahmini/
│
├── dashboard.py
├── requirements.txt
│
├── data/
│   ├── external/
│   │   └── 3year.arff
│   └── raw/
│       └── data.csv
│
├── models/
│   ├── best_xgb_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── Sirket_Iflas_Tahmini_Analizi.ipynb
│
├── results/
│   └── model_results.txt
│
└── src/
    ├── data_preprocessing.py
    ├── evaluate_models.py
    ├── predict.py
    ├── save_scaler.py
    └── train_models.py
```
---

## 🛠️ Kullanılan Teknolojiler
Python

Pandas

NumPy

Scikit-learn

XGBoost

Joblib

Matplotlib

Streamlit

Jupyter Notebook

Git & GitHub

---

## 📊 Model Değerlendirme
Model değerlendirmesinde özellikle Accuracy yerine sınıflandırma probleminin yapısına uygun metrikler dikkate alınmıştır.
Recall, gerçek iflas eden şirketlerin ne kadarının doğru şekilde tespit edildiğini gösterdiği için bu proje açısından önemli bir metriktir.
Precision, modelin iflas edeceğini tahmin ettiği şirketlerin ne kadarının gerçekten iflas ettiğini gösterir.
F1-Score, Precision ve Recall arasındaki dengeyi ölçer.
ROC-AUC, modelin iki sınıfı ayırt etme kabiliyetini değerlendirirken;
PR-AUC, özellikle sınıf dengesizliği bulunan veri setlerinde modelin azınlık sınıfındaki performansını değerlendirmek açısından önemlidir.
