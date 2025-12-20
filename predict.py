import joblib
from preprocessing import clean_text
from similarity import compute_similarity


# Load trained model, vectorizer and reference vectors
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
real_news_vectors = joblib.load("real_news_vectors.pkl")


def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    
    similarity_score = compute_similarity(vector, real_news_vectors)

    if prediction[0] == 1 and similarity_score > 0.6:
        return "REAL NEWS (High Confidence)"
    elif prediction[0] == 0 and similarity_score < 0.3:
        return "FAKE NEWS (Low Similarity to Trusted Sources)"
    else:
        return "UNCERTAIN — Requires Manual Verification"


# User input
news = input("Enter news text: ")
result = predict_news(news)
print("Prediction:", result)

