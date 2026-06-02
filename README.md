# 🛡️ Spam Email Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-98.30%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end **NLP-based Machine Learning project** that classifies SMS/email messages as **Spam or Legitimate (Ham)** using TF-IDF vectorization and Logistic Regression — deployed as an interactive Streamlit web application.

---

## 🌐 Live Demo

👉 **[Click here to try the live app](YOUR_STREAMLIT_URL_HERE)**

---

## 📸 Screenshots

> *(Add screenshots after deploying — drag images into this README on GitHub)*

---

## 📌 Project Overview

| Item           | Details                                  |
| -------------- | ---------------------------------------- |
| **Problem**    | Binary text classification (Spam vs Ham) |
| **Dataset**    | UCI SMS Spam Collection (5,572 messages) |
| **Best Model** | Logistic Regression + GridSearchCV       |
| **Accuracy**   | 98.30%                                   |
| **ROC-AUC**    | 0.9878                                   |
| **Deployment** | Streamlit Cloud                          |

---

## 🛠️ Tech Stack

| Component          | Technology               |
| ------------------ | ------------------------ |
| Language           | Python 3.12              |
| ML Framework       | scikit-learn 1.7.2       |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Data Processing    | pandas, numpy            |
| Visualization      | matplotlib, seaborn      |
| NLP                | TF-IDF Vectorizer        |
| Deployment         | Streamlit                |
| Model Saving       | joblib                   |

---

## 📁 Project Structure

```
Spam-Email-Detection/
│
├── app.py                          # Streamlit web application
├── spam_detector.pkl               # Trained & saved ML model
├── spam.csv                        # SMS Spam Collection dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
└── notebooks/
    └── spam_detection_kaggle.ipynb # Complete Kaggle notebook
```

---

## 🔁 ML Pipeline

```
Raw Text
↓
Text Preprocessing
(lowercase → remove URLs → remove emails → remove punctuation)
↓
TF-IDF Vectorization
(5000 features · unigrams + bigrams · log normalization)
↓
Class Balancing with SMOTE
(598 spam → 3859 synthetic spam samples)
↓
Model Training
(Naive Bayes + Logistic Regression)
↓
Hyperparameter Tuning
(GridSearchCV · 5-fold Cross Validation · 60 fits)
↓
Best Model Saved → spam_detector.pkl
```

---

## 📊 Results

| Model                         | Accuracy   | ROC-AUC    | Spam Recall |
| ----------------------------- | ---------- | ---------- | ----------- |
| Naive Bayes (baseline)        | 98.12%     | 0.9886     | 87%         |
| Logistic Regression + SMOTE   | 97.94%     | 0.9867     | 94%         |
| **LR Tuned — GridSearchCV ✅** | **98.30%** | **0.9878** | **88%**     |

### Best Hyperparameters (GridSearchCV)

```python
{
  'clf__C': 10.0,
  'tfidf__max_features': 5000,
  'tfidf__ngram_range': (1, 2),
  'tfidf__sublinear_tf': True
}
```

### 5-Fold Cross-Validation

| Model       | CV Accuracy    | CV F1-Score     |
| ----------- | -------------- | --------------- |
| Naive Bayes | 98.38% ± 0.22% | 0.9387 ± 0.0086 |
| LR + SMOTE  | 95.56% ± 0.28% | 0.9543 ± 0.0029 |

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Aastha-Sharma07/Spam-Email-Detection.git
cd Spam-Email-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. Open in browser

http://localhost:8501

---

## 📦 Dataset

* **Name:** SMS Spam Collection Dataset
* **Source:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
* **Size:** 5,572 messages
* **Classes:** Ham (86.6%) · Spam (13.4%)
* **Split:** 80% train / 20% test (stratified)

---

## 🔑 Key Features

* ✅ Real-time spam detection with confidence score
* ✅ NLP preprocessing pipeline (URLs, emails, punctuation removal)
* ✅ TF-IDF with bigrams for better phrase detection
* ✅ SMOTE to handle class imbalance
* ✅ GridSearchCV hyperparameter tuning (60 fits)
* ✅ 5-fold cross-validation for reliable evaluation
* ✅ Confusion matrix + ROC curve visualization
* ✅ Model saved with joblib for production deployment
* ✅ Interactive Streamlit web app

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Aastha Sharma**


---

