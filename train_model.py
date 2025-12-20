import pandas as pd
from preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy as np
from scipy.sparse import hstack   #merge sparse + dense features
from features import extract_metadata_features


# Load the dataset
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

print(fake.head())

print(real.head())

fake['label'] = 0
real['label'] = 1

# Combine the datasets and shuffle
data = pd.concat([fake,real])
data = data.sample(frac=1).reset_index(drop=True)

# Clean the text data
data['text'] = data['text'].apply(clean_text)

# Vectorize the text data to number using TF-IDF
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
X = vectorizer.fit_transform(data['text'])
y = data['label']

#create reference corpus using only real news
real_news_texts = data[data['label'] == 1]['text']
real_news_vectors = vectorizer.transform(real_news_texts)

# Extract metadata features
metadata_features = data["text"].apply(extract_metadata_features)
metadata_features = np.array(metadata_features.tolist())
#X-TF-IDF features (sparse matrix)
#metadata_features (dense matrix)

#combine text features and metadata features
X_combined = hstack([X, metadata_features])

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#test the model
y_pred = model.predict(X_test)

#accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
#I analyzed false positives and false negatives to understand model failure modes

#evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save the trained model and vectorizer
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(real_news_vectors, "real_news_vectors.pkl") 

print("Model and vectorizer saved.")