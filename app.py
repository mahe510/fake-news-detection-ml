import streamlit as st
import joblib

from preprocessing import clean_text
from features import extract_metadata_features
from similarity import compute_similarity
import numpy as np
from scipy.sparse import hstack

# Load saved components
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
real_news_vectors = joblib.load("real_news_vectors.pkl")

st.title("📰 Fake News Detection Demo")

text = st.text_area("Enter news text here:")

if st.button("Check News"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess
        cleaned = clean_text(text)

        # TF-IDF
        text_vector = vectorizer.transform([cleaned])

        # Metadata
        meta = extract_metadata_features(text)
        meta = np.array(meta).reshape(1, -1)

        # Feature fusion
        combined = hstack([text_vector, meta])

        # Prediction
        prediction = model.predict(combined)[0]

        # Similarity
        similarity = compute_similarity(text_vector, real_news_vectors)

        # Final decision
        if prediction == 1 and similarity > 0.6:
            st.success("✅ REAL NEWS (High confidence)")
        elif prediction == 0 and similarity < 0.3:
            st.error("❌ FAKE NEWS (Low similarity to trusted sources)")
        else:
            st.warning("⚠️ UNCERTAIN — Manual verification recommended")
