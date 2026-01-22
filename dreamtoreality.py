import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

# ------------------ UI CONFIG -------------------

st.set_page_config(page_title="Dream to Reality", layout="centered")

# Colorful animated gradient background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 30%, #fbc2eb 60%, #a18cd1 100%);
        background-attachment: fixed;
    }
    textarea {
        border-radius: 12px !important;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #a18cd1;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ BACKEND ---------------------

SMALL_STOPWORDS = set("""
a an the and is in to of for on it i was my we me you your our they them this that with
""".split())

POSITIVE = {"joy","happy","laugh","love","peaceful","festival","smile","free","delight","fun","joyful","relief"}
NEGATIVE = {"fear","fearful","scared","chase","chased","fall","falling","anxiety","stress","missed","late","dark","whisper","panic"}

def clean_text_simple(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def simple_tokenize(text):
    t = clean_text_simple(text)
    return [w for w in re.findall(r"[a-z]{2,}", t) if w not in SMALL_STOPWORDS]

def lexicon_emotion_features(clean_text):
    toks = clean_text.split()
    pos = sum(1 for t in toks if t in POSITIVE)
    neg = sum(1 for t in toks if t in NEGATIVE)
    total = max(len(toks), 1)
    return {"pos_ratio": pos/total, "neg_ratio": neg/total}

@st.cache_resource
def build_pipeline():
    data = {
        "dream": [
            "I was falling endlessly and couldn't stop.",
            "I met my grandmother and we hugged and laughed.",
            "I missed the train even though I ran fast.",
            "I was flying above mountains with great joy.",
            "I was chased and couldn't find a door.",
            "I found my childhood home and felt peaceful.",
            "I kept being late for an exam and couldn't write.",
            "A bright festival with family and music.",
            "Walking alone in a dark forest and hearing whispers.",
            "I discovered a hidden room with old toys."
        ],
        "mood_score": [0.2,0.9,0.4,0.95,0.25,0.85,0.3,0.92,0.15,0.78]
    }

    df = pd.DataFrame(data)
    df["clean"] = df["dream"].apply(lambda x: " ".join(simple_tokenize(x)))
    emo = df["clean"].apply(lexicon_emotion_features).apply(pd.Series)
    df = pd.concat([df, emo], axis=1)

    vect = CountVectorizer(max_features=1000)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    topics = lda.fit_transform(vect.fit_transform(df["clean"]))
    topic_cols = [f"topic_{i}" for i in range(3)]
    df = pd.concat([df, pd.DataFrame(topics, columns=topic_cols)], axis=1)

    tfidf = TfidfVectorizer(max_features=300)
    X_text = tfidf.fit_transform(df["clean"])
    X_tab = df[["pos_ratio","neg_ratio"] + topic_cols].values
    X_full = hstack([X_text, csr_matrix(X_tab)])
    y = df["mood_score"].values

    reg = RandomForestRegressor(n_estimators=150, random_state=42)
    reg.fit(X_full, y)

    return vect, lda, tfidf, reg, topic_cols


# ------------------ FRONTEND ---------------------

st.title("Dream to Reality – Emotional State Predictor")

st.caption("Enter your dream below. The system will predict your emotional state only.")

with st.spinner("Training model..."):
    vect, lda, tfidf, reg, topic_cols = build_pipeline()

dream_input = st.text_area("💭 Describe your dream:", height=140)

if st.button("Predict Emotional State"):
    if not dream_input.strip():
        st.warning("Please enter a dream first!")
    else:
        clean = " ".join(simple_tokenize(dream_input))
        lex = lexicon_emotion_features(clean)

        dtm = vect.transform([clean])
        topic_new = lda.transform(dtm)[0]
        tfidf_new = tfidf.transform([clean])

        tab_vec = np.hstack([lex["pos_ratio"], lex["neg_ratio"], topic_new])
        X_new = hstack([tfidf_new, csr_matrix(tab_vec.reshape(1, -1))])

        pred = float(reg.predict(X_new)[0])

        def mood_label(score):
            if score >= 0.85: return "Very Positive / Happy"
            if score >= 0.65: return "Positive / Calm"
            if score >= 0.45: return "Neutral"
            if score >= 0.25: return "Stressed / Anxious"
            return "Negative / Distressed"

        mood_state = mood_label(pred)

        # ---------- ONLY DISPLAY EMOTIONAL STATE ----------
        st.subheader("Emotional State")
        st.success(f"{mood_state}")
