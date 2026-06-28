import streamlit as st
import pickle
import re
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen BoP Indonesia",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# LOAD DEPENDENCIES (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_nltk():
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("indonesian"))

@st.cache_resource
def load_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource
def load_kamus():
    url = "https://raw.githubusercontent.com/hilmiammar/analisis-sentimen/master/kamuskatabaku.xlsx"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        kamus_data = pd.read_excel(BytesIO(response.content))
        return dict(zip(kamus_data["non_standard"], kamus_data["standard_word"]))
    except Exception as e:
        st.warning(f"Gagal memuat kamus tidak baku: {e}. Normalisasi dilewati.")
        return {}

@st.cache_resource
def load_model_and_vectorizer():
    with open("multinomial_nb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# ─────────────────────────────────────────────
# PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────

def case_folding(text):
    return text.lower() if isinstance(text, str) else text

def remove_URL(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text) if isinstance(text, str) else text

def remove_html(text):
    return re.compile(r"<.*?>").sub("", text) if isinstance(text, str) else text

def remove_emoji(text):
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)

def remove_symbols(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text) if isinstance(text, str) else text

def remove_numbers(text):
    return re.sub(r"\d", "", text) if isinstance(text, str) else text

def remove_usernames(text):
    return re.sub(r"@\w+", "", text) if isinstance(text, str) else text

def remove_hashtags(text):
    return re.sub(r"#\w+", "", text) if isinstance(text, str) else text

def replace_taboo_words(text, kamus):
    if not isinstance(text, str):
        return text
    words = text.split()
    return " ".join([kamus.get(w, w) for w in words])

CUSTOM_STOPWORDS = {
    "rt","user","username","url","https","t","co","link","via","dm","amp","tag","hastag",
    "nih","dong","sih","tuh","deh","lah","kok","kan","ya","aja","gitu","gini","kayak","kaya",
    "wkwk","wkwkwk","haha","hehe","wk","mah","saya","gue","gw","gua","lo","lu",
    "kak","bang","mas","mbak","kakak","adik",
}

def preprocess(text, kamus, stop_words, stemmer):
    text = case_folding(text)
    text = remove_URL(text)
    text = remove_usernames(text)
    text = remove_hashtags(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_symbols(text)
    text = remove_numbers(text)
    text = replace_taboo_words(text, kamus)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and w not in CUSTOM_STOPWORDS]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

def predict_sentiment(text, model, vectorizer, kamus, stop_words, stemmer):
    cleaned = preprocess(text, kamus, stop_words, stemmer)
    if not cleaned.strip():
        return None, cleaned
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    return pred, cleaned, dict(zip(classes, proba))

# ─────────────────────────────────────────────
# SENTIMENT STYLING
# ─────────────────────────────────────────────

SENTIMENT_EMOJI = {"Positif": "😊", "Netral": "😐", "Negatif": "😠"}
SENTIMENT_COLOR = {"Positif": "#28a745", "Netral": "#6c757d", "Negatif": "#dc3545"}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.title("⚙️ Pengaturan")
menu = st.sidebar.radio(
    "Pilih Fitur:",
    ["🏠 Beranda", "🔍 Klasifikasi Teks", "📁 Klasifikasi File CSV"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "Model: **Multinomial Naive Bayes with SMOTE**\n\n"
    "Vectorizer: **TF-IDF**\n\n"
    "Preprocessing: Case Folding → Cleaning → Normalisasi → Tokenisasi → Stopword → Stemming"
)

# ─────────────────────────────────────────────
# LOAD ALL RESOURCES
# ─────────────────────────────────────────────

with st.spinner("Memuat model dan sumber daya..."):
    try:
        model, vectorizer = load_model_and_vectorizer()
        stop_words_base = load_nltk()
        stop_words = stop_words_base.union(CUSTOM_STOPWORDS)
        stemmer = load_stemmer()
        kamus = load_kamus()
        resources_ok = True
    except FileNotFoundError as e:
        resources_ok = False
        missing = str(e)

# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

if menu == "🏠 Beranda":
    st.title("📊 Analisis Sentimen BoP Indonesia")
    st.markdown(
        """
        Aplikasi ini melakukan **klasifikasi sentimen** teks Bahasa Indonesia
        menggunakan model **Multinomial Naive Bayes with SMOTE** yang dilatih dengan data
        Board of Peace (BoP) Indonesia.
        
        ### Kelas Sentimen
        | Label | Arti |
        |-------|------|
        | 😊 Positif | Teks mengandung sentimen positif |
        | 😐 Netral | Teks bersifat netral / informatif |
        | 😠 Negatif | Teks mengandung sentimen negatif |

        ### Alur Pipeline
        ```
        Input Teks
            ↓  Preprocessing (Case Folding → Cleaning → Normalisasi → Tokenisasi → Stopword → Stemming)
            ↓  TF-IDF Vectorization
            ↓  Multinomial NB with SMOTE
            ↓  Klasifikasi
        Output Sentimen
        ```
        """
    )

    if not resources_ok:
        st.error(
            f"⚠️ File model tidak ditemukan: `{missing}`\n\n"
            "Pastikan file `multinomial_nb_model.pkl` dan `tfidf_vectorizer.pkl` "
            "berada di direktori yang sama dengan `app.py`."
        )

elif menu == "🔍 Klasifikasi Teks":
    st.title("🔍 Klasifikasi Sentimen Teks")

    if not resources_ok:
        st.error(f"⚠️ Model tidak tersedia: {missing}")
    else:
        user_input = st.text_area(
            "Masukkan teks yang ingin diklasifikasi:",
            placeholder="Contoh: Indonesia gabung BoP merupakan langkah positif untuk perdamaian.",
            height=150,
        )

        col1, col2 = st.columns([1, 4])
        predict_btn = col1.button("🔮 Klasifikasi", use_container_width=True)
        col2.button("🗑️ Bersihkan", on_click=lambda: None, use_container_width=True)

        if predict_btn:
            if not user_input.strip():
                st.warning("Silakan masukkan teks terlebih dahulu.")
            else:
                with st.spinner("Memproses..."):
                    result = predict_sentiment(user_input, model, vectorizer, kamus, stop_words, stemmer)

                if result[0] is None:
                    st.error("Teks tidak dapat diproses (kosong setelah preprocessing).")
                else:
                    label, cleaned_text, probas = result
                    emoji = SENTIMENT_EMOJI[label]
                    color = SENTIMENT_COLOR[label]

                    st.markdown(
                        f"""
                        <div style="
                            background-color:{color}22;
                            border-left: 5px solid {color};
                            padding: 20px;
                            border-radius: 8px;
                            margin: 10px 0;
                        ">
                            <h2 style="color:{color}; margin:0;">{emoji} {label}</h2>
                            <p style="margin:5px 0 0 0; color:#555;">Klasifikasi sentimen untuk teks yang dimasukkan</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Confidence bar chart
                    st.markdown("#### Tingkat Kepercayaan Model")
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    labels_sorted = ["Positif", "Netral", "Negatif"]
                    values = [probas.get(l, 0) for l in labels_sorted]
                    colors = [SENTIMENT_COLOR[l] for l in labels_sorted]
                    bars = ax.barh(labels_sorted, values, color=colors)
                    for bar, val in zip(bars, values):
                        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                                f"{val:.1%}", va="center", fontsize=10)
                    ax.set_xlim(0, 1.15)
                    ax.set_xlabel("Probabilitas")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)

                    with st.expander("📋 Detail Preprocessing"):
                        st.markdown(f"**Teks asli:**\n> {user_input}")
                        st.markdown(f"**Teks setelah preprocessing:**\n> `{cleaned_text}`")

elif menu == "📁 Klasifikasi File CSV":
    st.title("📁 Klasifikasi Batch dari File CSV")

    if not resources_ok:
        st.error(f"⚠️ Model tidak tersedia: {missing}")
    else:
        st.markdown(
            "Upload file CSV dengan kolom teks. Aplikasi akan menambahkan kolom `Sentimen_Klasifikasi`."
        )

        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
            st.markdown(f"**{len(df_input)} baris ditemukan.** Preview:")
            st.dataframe(df_input.head())

            text_col = st.selectbox("Pilih kolom teks:", df_input.columns.tolist())

            if st.button("🚀 Jalankan Klasifikasi Batch"):
                progress = st.progress(0)
                predictions = []
                total = len(df_input)

                for i, text in enumerate(df_input[text_col].astype(str)):
                    result = predict_sentiment(text, model, vectorizer, kamus, stop_words, stemmer)
                    predictions.append(result[0] if result[0] else "Tidak Dapat Diklasifikasi")
                    progress.progress((i + 1) / total)

                df_input["Sentimen_Klasifikasi"] = predictions
                st.success("✅ Klasifikasi selesai!")
                st.dataframe(df_input)

                # Distribusi hasil
                st.markdown("#### Distribusi Hasil Klasifikasi")
                counts = df_input["Sentimen_Klasifikasi"].value_counts()
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.bar(
                    counts.index,
                    counts.values,
                    color=[SENTIMENT_COLOR.get(l, "#999") for l in counts.index],
                )
                for i, (_, v) in enumerate(counts.items()):
                    ax.text(i, v + 0.5, str(v), ha="center")
                ax.set_xlabel("Sentimen")
                ax.set_ylabel("Jumlah")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

                # Download
                csv_out = df_input.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "⬇️ Download Hasil CSV",
                    csv_out,
                    file_name="hasil_klasifikasi_sentimen.csv",
                    mime="text/csv",
                )