import os
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Şirket İflas Tahmini",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# DOSYA YOLLARI VE SABİTLER
# ============================================================

DATA_PATH = "data/raw/data.csv"
MODEL_PATH = "models/best_xgb_model.pkl"
SCALER_PATH = "models/scaler.pkl"

TARGET = "Bankrupt?"
THRESHOLD = 0.40

# ============================================================
# SELECTBOX VE TÜM COMPONENTLERİ SİYAH YAPAN ENJEKSİYON
# ============================================================

st.markdown(
    """
    <style>
    /* Ana Ekran Arka Planı */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0b0b0d !important;
    }

    /* SELECTBOX KUTUSUNU SİYAH VE YAZIYI BEYAZ YAPMA (GÖRSELDEKİ BEYAZ KUTU) */
    div[data-baseweb="select"] > div:first-child {
        background-color: #121215 !important;
        border: 1px solid #33333b !important;
        border-radius: 8px !important;
    }

    /* Kutu İçindeki "Company Record #5" Yazısı ve Ok Simgesi */
    div[data-baseweb="select"] div, 
    div[data-baseweb="select"] span, 
    div[data-baseweb="select"] p,
    div[data-baseweb="select"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* Açılan Dropdown Liste Menüsü Arka Planı ve Yazıları */
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] div,
    ul[data-baseweb="menu"] {
        background-color: #121215 !important;
    }

    /* Seçenek Yazıları */
    li[data-baseweb="option"] span,
    li[data-baseweb="option"] div {
        color: #ffffff !important;
    }

    /* Seçenek Üzerine Gelindiğinde (Hover) Arka Plan */
    li[data-baseweb="option"]:hover {
        background-color: #22222a !important;
    }

    /* Başlıklar ve Genel Metinler */
    .custom-title {
        font-size: 38px;
        font-weight: 700;
        color: #ffffff !important;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .custom-subtitle {
        font-size: 15px;
        color: #a0a0ab !important;
        margin-bottom: 30px;
    }

    .custom-section {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff !important;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 1px solid #222226;
        padding-bottom: 8px;
    }

    /* Kart Tasarımları */
    .result-card {
        background-color: #121215;
        border: 1px solid #33333b;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 25px;
    }

    .result-label {
        color: #a0a0ab !important;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .probability {
        font-size: 44px;
        font-weight: 700;
        color: #ffffff !important;
        margin-top: 5px;
    }

    .threshold-text {
        color: #85858c !important;
        font-size: 12px;
        margin-top: 8px;
    }

    .high-risk {
        color: #ff5555 !important;
    }

    .low-risk {
        color: #50fa7b !important;
    }

    .metric-box {
        background-color: #121215;
        border: 1px solid #29292e;
        border-radius: 12px;
        padding: 18px 10px;
        text-align: center;
    }

    .metric-name {
        color: #a0a0ab !important;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .metric-value {
        color: #ffffff !important;
        font-size: 24px;
        font-weight: 700;
        margin-top: 6px;
    }

    .feature-box {
        background-color: #121215;
        border: 1px solid #252529;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    .feature-name {
        color: #a0a0ab !important;
        font-size: 12px;
    }

    .feature-value {
        color: #ffffff !important;
        font-size: 18px;
        font-weight: 600;
        margin-top: 5px;
    }

    .info-box {
        background-color: #121215;
        border-left: 4px solid #6272a4;
        border-radius: 6px;
        padding: 14px 17px;
        color: #ffffff !important;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .footer-text {
        text-align: center;
        color: #55555c !important;
        font-size: 12px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #222226;
    }

    /* Proje Hakkında kutusundaki yazıyı beyaz yap */
    div[data-testid="stAlert"] {
    background-color: #121215 !important;
    border: 1px solid #29292e !important;
    }

    div[data-testid="stAlert"] p {
    color: #ffffff !important;
    }

    .project-info {
    background-color: #121215;
    border: 1px solid #29292e;
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 45px;
    margin-bottom: 25px;
    color: #a0a0ab !important;
    font-size: 13px;
    line-height: 1.7;
}

.project-info-title {
    color: #ffffff !important;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 8px;
}

.project-info-text {
    color: #85858c !important;
}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DOSYA VE MODEL KONTROLLERİ
# ============================================================

if not os.path.exists(DATA_PATH):
    st.error(f"Veri seti bulunamadı: {DATA_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model bulunamadı: {MODEL_PATH}")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error(f"Scaler bulunamadı: {SCALER_PATH}")
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    df = load_data()
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Dosyalar yüklenirken hata oluştu: {e}")
    st.stop()

if hasattr(scaler, "feature_names_in_"):
    FEATURES = list(scaler.feature_names_in_)
else:
    FEATURES = [col for col in df.columns if col != TARGET]

# ============================================================
# BAŞLIK
# ============================================================

st.markdown('<div class="custom-title">Şirket İflas Tahmini</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">XGBoost tabanlı şirket iflas risk değerlendirme sistemi</div>', unsafe_allow_html=True)

# ============================================================
# ŞİRKET SEÇİMİ (AÇILIR MENÜ)
# ============================================================

st.markdown('<div class="custom-section">Şirket / Kayıt Seçimi</div>', unsafe_allow_html=True)

company_options = list(range(len(df)))
selected_index = st.selectbox(
    "Analiz edilecek veri kayıt numarasını seçin:",
    company_options,
    format_func=lambda x: f"Company Record #{x + 1}"
)

selected_row = df.iloc[selected_index]

# ============================================================
# TEMEL FİNANSAL GÖSTERGELER
# ============================================================

st.markdown('<div class="custom-section">Temel Finansal Göstergeler</div>', unsafe_allow_html=True)

important_features = [
    ("ROA(C) before interest and depreciation before terminating quantities", "Kârlılık (ROA)"),
    ("Debt ratio %", "Borçluluk Oranı"),
    ("Net worth/Assets", "Özkaynak Yapısı"),
    ("Interest Expense Ratio", "Faiz Yükü")
]

indicator_columns = st.columns(4)

for col, (feat_name, label) in zip(indicator_columns, important_features):
    with col:
        val = selected_row.get(feat_name, selected_row[FEATURES[indicator_columns.index(col)]])
        st.markdown(
            f"""
            <div class="feature-box">
                <div class="feature-name">{label}</div>
                <div class="feature-value">{float(val):.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# TAHMİN BÖLÜMÜ
# ============================================================

st.write("")
st.write("")

if st.button("TAHMİN HESAPLA", use_container_width=True, type="primary"):
    try:
        X_company = selected_row[FEATURES].to_frame().T
        X_company = X_company.apply(pd.to_numeric, errors="coerce")

        X_company_scaled = scaler.transform(X_company)
        probability = model.predict_proba(X_company_scaled)[0, 1]
        prediction = int(probability >= THRESHOLD)

        # ----------------------------------------------------
        # MODEL TAHMİN SONUCU
        # ----------------------------------------------------
        st.markdown('<div class="custom-section">Model Tahmini</div>', unsafe_allow_html=True)

        if prediction == 1:
            result_text = "⚠ İFLAS RİSKİ YÜKSEK"
            result_class = "high-risk"
        else:
            result_text = "✓ İFLAS RİSKİ DÜŞÜK"
            result_class = "low-risk"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Model Prediction</div>
                <div class="result-title {result_class}">{result_text}</div>
                <div class="result-label">Tahmin Edilen İflas Olasılığı</div>
                <div class="probability">%{probability * 100:.2f}</div>
                <div class="threshold-text">Classification threshold: {THRESHOLD:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # FEATURE TABLOSU (SİYAH ARKA PLAN - BEYAZ METİN)
        # ----------------------------------------------------
        st.markdown(f'<div class="custom-section">Seçilen Şirketin {len(FEATURES)} Feature Değeri</div>', unsafe_allow_html=True)

        feature_table = pd.DataFrame({
            "Feature": FEATURES,
            "Value": [selected_row[f] for f in FEATURES]
        })
        feature_table["Value"] = pd.to_numeric(feature_table["Value"], errors="coerce")

        styled_df = feature_table.style.set_properties(**{
            'background-color': '#0b0b0d',
            'color': '#ffffff',
            'border-color': '#222226'
        }).set_table_styles([
            {
                'selector': 'th',
                'props': [('background-color', '#121215'), ('color', '#ffffff'), ('font-weight', 'bold')]
            }
        ])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error("Tahmin sırasında bir hata oluştu.")
        st.exception(e)


# ============================================================
# PROJE HAKKINDA
# ============================================================

st.markdown(
    '<div class="custom-section">Proje Hakkında</div>',
    unsafe_allow_html=True
)

st.info(
    "Bu dashboard, şirketlerin finansal göstergelerini "
    "kullanarak iflas riskini tahmin etmek amacıyla geliştirilmiştir. "
    "XGBoost modeli, şirketlerin 95 finansal özelliğini analiz ederek "
    "iflas olasılığı üretmektedir. Model performansı Accuracy, "
    "Precision, Recall, F1-Score, ROC-AUC ve PR-AUC metrikleri "
    "kullanılarak değerlendirilmiştir."
)



# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        Company Bankruptcy Prediction · XGBoost · Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)