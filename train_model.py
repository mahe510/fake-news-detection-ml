print("Script started...")

import pandas as pd
import numpy as np
import torch
import joblib
import os
import matplotlib.pyplot as plt

from preprocessing import clean_text
from features import extract_metadata_features

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from scipy.sparse import hstack
from transformers import DistilBertTokenizer, DistilBertModel


# ===============================
# 1. Load Dataset
# ===============================
print("Loading Fake.csv...")
fake = pd.read_csv("data/Fake.csv")
print("Fake.csv loaded.")

print("Loading True.csv...")
real = pd.read_csv("data/True.csv")
print("True.csv loaded.")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text
print("Cleaning text...")
data["text"] = data["text"].apply(clean_text)

X_text = data["text"]
y = data["label"]


# ===============================
# 2. Train-Test Split (NO LEAKAGE)
# ===============================
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print("Train-test split complete.")


# ===============================
# 3. TF-IDF
# ===============================
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print("TF-IDF complete.")


# ===============================
# 4. Metadata Features
# ===============================
meta_train = np.array(X_train_text.apply(extract_metadata_features).tolist())
meta_test = np.array(X_test_text.apply(extract_metadata_features).tolist())

print("Metadata extraction complete.")


# ===============================
# 5. DistilBERT Embeddings (CACHED + BATCHED)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_file_train = "bert_train.npy"
embedding_file_test = "bert_test.npy"

if os.path.exists(embedding_file_train) and os.path.exists(embedding_file_test):
    print("Loading cached BERT embeddings...")
    bert_train = np.load(embedding_file_train)
    bert_test = np.load(embedding_file_test)

else:
    print("Generating DistilBERT embeddings (one-time process)...")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    bert_model.to(device)
    bert_model.eval()

    def get_embeddings(text_series, batch_size=32):
        embeddings = []
        texts = list(text_series)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            print(f"Processing batch {i} / {len(texts)}")

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = bert_model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.extend(cls_embeddings.cpu().numpy())

        return np.array(embeddings)

    bert_train = get_embeddings(X_train_text)
    bert_test = get_embeddings(X_test_text)

    np.save(embedding_file_train, bert_train)
    np.save(embedding_file_test, bert_test)

    print("Embeddings saved for future fast runs.")


# ===============================
# 6. Combine Features
# ===============================
X_train_combined = hstack([X_train_tfidf, meta_train, bert_train])
X_test_combined = hstack([X_test_tfidf, meta_test, bert_test])

X_train_combined = X_train_combined.tocsr()
X_test_combined = X_test_combined.tocsr()


print("Feature fusion complete.")


# ===============================
# 7. Train Final Model
# ===============================
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train_combined, y_train)

print("Model training complete.")


# ===============================
# 8. Evaluation on Test Set
# ===============================
y_pred = model.predict(X_test_combined)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# ===============================
# 9. ROC-AUC
# ===============================
y_prob = model.predict_proba(X_test_combined)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", auc_score)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()


# ===============================
# 10. 5-Fold Cross Validation
# ===============================
print("\nRunning 5-Fold Cross Validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracies = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train_combined, y_train)):
    
    X_fold_train = X_train_combined[train_index]
    X_fold_val = X_train_combined[val_index]
    
    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]
    
    fold_model = LogisticRegression(max_iter=1000, solver="liblinear")
    fold_model.fit(X_fold_train, y_fold_train)
    
    y_fold_pred = fold_model.predict(X_fold_val)
    acc = accuracy_score(y_fold_val, y_fold_pred)
    
    cv_accuracies.append(acc)
    
    print(f"Fold {fold+1} Accuracy: {acc:.4f}")

print("\nCross Validation Results:")
print("Mean Accuracy:", np.mean(cv_accuracies))
print("Std Deviation:", np.std(cv_accuracies))


# ===============================
# 11. Save Model
# ===============================
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved.")
