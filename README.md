# Context-Aware Fake News Detection Using Social Media Data

## Overview
This project implements an end-to-end Machine Learning system to detect and flag fake news on social media platforms.  
Unlike basic text classifiers, the system combines **linguistic patterns, metadata features, and content similarity verification** to make robust and confidence-aware predictions.

The goal is to demonstrate **real-world ML system design**, not just model accuracy.

---

## Key Features
- NLP-based text preprocessing
- TF-IDF feature extraction with n-grams
- Metadata feature engineering (emotion & style signals)
- Feature fusion (text + metadata)
- Content similarity verification using cosine similarity
- Confidence-aware fake/real classification
- Error-aware design (handles uncertainty cases)

---

## System Architecture

Social Media Post / News Article
↓
Text Cleaning & Normalization
↓
Feature Extraction Layer
├── TF-IDF (word & phrase patterns)
├── Metadata Features (emotion & style)
↓
Feature Fusion
↓
Fake News Classifier (Logistic Regression)
↓
Content Similarity Verification
↓
Final Decision + Confidence Level



---

## ⚙️ Features Implemented

### 1️⃣ Text-Based Features
- TF-IDF vectorization
- Unigrams and bigrams
- Captures important linguistic patterns

### 2️⃣ Metadata Features
- Text length
- Capital letter ratio
- Exclamation mark count
- Lexical diversity

### 3️⃣ Content Similarity
- Compares input text with trusted real news
- Uses cosine similarity
- Helps detect suspicious or novel content

---

## 📊 Model Performance
- Accuracy: **~98.9%**
- Evaluated using:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

Error analysis was performed to understand false positives and false negatives.

---

## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- NLP (TF-IDF, cosine similarity)

---


---

## ⚙️ Features Implemented

### 1️⃣ Text-Based Features
- TF-IDF vectorization
- Unigrams and bigrams
- Captures important linguistic patterns

### 2️⃣ Metadata Features
- Text length
- Capital letter ratio
- Exclamation mark count
- Lexical diversity

### 3️⃣ Content Similarity
- Compares input text with trusted real news
- Uses cosine similarity
- Helps detect suspicious or novel content

---

## 📊 Model Performance
- Accuracy: **~98.9%**
- Evaluated using:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

Error analysis was performed to understand false positives and false negatives.

---

## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- NLP (TF-IDF, cosine similarity)

---

## ▶️ How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
