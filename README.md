# 📰 Context-Aware Fake News Detection Using Hybrid NLP & Deep Learning

## 📌 Overview

This project implements an end-to-end Machine Learning system to detect and flag fake news using a hybrid architecture combining:

- Classical NLP techniques
- Metadata feature engineering
- Transformer-based contextual embeddings
- Content similarity verification

Unlike basic text classifiers, this system is designed to simulate real-world ML pipeline engineering, focusing on robustness, stability, and interpretability.

---

## 🚀 Key Highlights

- Hybrid Feature Architecture (TF-IDF + Metadata + DistilBERT)
- Contextual semantic understanding using Transformer embeddings
- Content similarity verification with trusted news corpus
- Confidence-aware classification
- 5-Fold Stratified Cross Validation
- ROC-AUC evaluation
- Streamlit-based interactive UI

---

## 🧠 System Architecture

News Article / Social Media Post  
↓  
Text Cleaning & Normalization  
↓  
Feature Extraction Layer  
├── TF-IDF (surface linguistic patterns)  
├── Metadata Features (style & behavior signals)  
├── DistilBERT Embeddings (contextual semantics)  
↓  
Feature Fusion (Sparse + Dense)  
↓  
Logistic Regression Classifier  
↓  
Content Similarity Verification  
↓  
Final Decision + Confidence Score  

---

## ⚙️ Feature Engineering

### 1️⃣ Text-Based Features (TF-IDF)

- Unigrams and Bigrams  
- 8000 max vocabulary features  
- Captures stylistic and lexical patterns  

### 2️⃣ Metadata Features

- Text length  
- Capital letter ratio  
- Exclamation mark count  
- Lexical diversity  

These features help detect emotionally manipulative or sensational writing styles.

### 3️⃣ Contextual Embeddings (DistilBERT)

- Transformer-based deep contextual representation  
- Captures semantic meaning beyond keyword frequency  
- Batch processing with caching for efficiency  

### 4️⃣ Content Similarity Verification

- Cosine similarity against trusted real-news corpus  
- Helps detect novel or suspicious content  
- Adds secondary verification layer  

---

## 📊 Model Evaluation

### Final Test Set Performance

- Accuracy: ~99%  
- ROC-AUC: ~0.999  
- Balanced precision and recall  

### Validation Strategy

- Stratified 5-Fold Cross Validation  
- Mean Accuracy: ~0.99  
- Low standard deviation (stable model)  

### Metrics Used

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- ROC Curve  
- AUC Score  

---

## 🛠️ Technologies Used

- Python  
- Scikit-learn  
- Pandas & NumPy  
- PyTorch  
- HuggingFace Transformers  
- Streamlit  
- TF-IDF & Cosine Similarity (NLP)  

---

## 💻 Project Structure

fake_news_detection/  
│  
├── app.py                  # Streamlit UI  
├── train_model.py          # Training pipeline  
├── preprocessing.py        # Text cleaning functions  
├── features.py             # Metadata feature extraction  
├── similarity.py           # Cosine similarity verification  
├── data/                   # Dataset (Fake & True news CSV)  
├── requirements.txt  
├── README.md  

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

pip install -r requirements.txt

### 2️⃣ Train the Model (Optional if model.pkl exists)

python train_model.py

### 3️⃣ Launch Web App

python -m streamlit run app.py

Then open:

http://localhost:8501

---
