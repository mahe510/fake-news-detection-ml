import streamlit as st
import joblib
import numpy as np
import torch

from preprocessing import clean_text
from features import extract_metadata_features
from similarity import compute_similarity
from transformers import DistilBertTokenizer, DistilBertModel
from scipy.sparse import hstack


# ===============================
# Load Saved Model & Components
# ===============================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
real_news_vectors = joblib.load("real_news_vectors.pkl")

# Load DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert_model.eval()


# ===============================
# Streamlit UI
# ===============================
st.title("📰 Context-Aware Fake News Detection")

text = st.text_area("Enter news text here:")

if st.button("Check News"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Clean
        cleaned = clean_text(text)

        # TF-IDF
        text_vector = vectorizer.transform([cleaned])

        # Metadata
        meta = extract_metadata_features(cleaned)
        meta = np.array(meta).reshape(1, -1)

        # BERT
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = bert_model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()

        # Combine ALL FEATURES
        combined = hstack([text_vector, meta, cls_embedding]).tocsr()

        # Debug (optional)
        # st.write("Combined shape:", combined.shape)

        # Prediction
        prediction = model.predict(combined)[0]
        probability = model.predict_proba(combined)[0][1]

        # Similarity
        similarity = compute_similarity(text_vector, real_news_vectors)

        if prediction == 1 and similarity > 0.6:
            st.success(f"✅ REAL NEWS ({probability:.2%} confidence)")
        elif prediction == 0 and similarity < 0.3:
            st.error(f"❌ FAKE NEWS ({1-probability:.2%} confidence)")
        else:
            st.warning("⚠️ UNCERTAIN — Manual verification recommended")

        st.info(f"Similarity Score: {similarity:.2f}")
