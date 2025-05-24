# 🏰 Sentiment Analysis of Disneyland Reviews

This project was developed as the Capstone for the Data Science Master's Program at IMMUNE Technology Institute. It applies Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning (DL) techniques to classify 42,000+ reviews from Disneyland parks (California, Paris, Hong Kong) into **positive** or **negative** sentiments.

## 📌 Objective

Analyze and classify guest reviews using a variety of ML and DL models to uncover actionable insights and patterns across different park locations and visitor profiles.

## 🧾 Dataset Overview

- **Total records**: 42,656 reviews  
- **Features**: `review_id`, `review_text`, `rating`, `year_month`, `reviewer_location`, `branch`  
- **Target variable**:  
  - `Positive` (rating ≥ 4) → class 1  
  - `Negative` (rating ≤ 3) → class 0  
- **Class distribution**: 91% positive, 9% negative

## 🧹 Data Preprocessing

- Standardized column names and date formats
- Removed missing or malformed entries
- Cleaned review texts (punctuation, stopwords, lemmatization)
- Tokenization and sequence padding for DL models
- Addressed class imbalance using:
  - `class_weight`
  - Focal Loss
  - Threshold tuning for rule-based models

## 🔍 Exploratory Data Analysis (EDA)

- **Histogram**: Ratings distribution (most users gave a 5)
- **Pie chart**: California park had the most reviews (45.5%)
- **Line chart**: Review volume peaked in 2015
- **Bar plots**: Top countries by review count (USA leads)
- **Sentiment bar chart**: Positive reviews dominate (≈75%)

## 🤖 Models Implemented

### 💡 Machine Learning

| Model               | Accuracy | AUC  | Notes |
|---------------------|----------|------|-------|
| Logistic Regression | 84%      | 0.90 | High precision on positives |
| Linear SVM (SVC)    | 90%      | 0.92 | Best performance overall |
| Naive Bayes         | 85%      | 0.90 | Strong on positives, weak on negatives |
| Decision Tree       | 77%      | 0.72 | Easily interpretable but low accuracy |
| XGBoost             | 85%      | 0.90 | Balanced, good F1 for negatives |

### 🧠 Deep Learning & NLP

| Model               | Accuracy | AUC  | Notes |
|---------------------|----------|------|-------|
| VADER (NLTK)        | 82%      | 0.70 | Rule-based, good for comparison |
| RNN (Keras)         | 79%      | 0.50 | Poor performance on negatives |
| LSTM (Bidirectional)| 89%      | 0.89 | Stronger than RNN, good F1 |
| BERT + LogisticReg  | 82%      | 0.89 | Fast embeddings, decent recall |
| **DistilBERT (Hugging Face)** | **90%** | **0.92** | 🚀 Best overall model |

## 📈 Evaluation Metrics

- Confusion Matrix
- Classification Report (Precision, Recall, F1)
- ROC-AUC Curve
- Precision-Recall Curve (Average Precision)

> Special attention was paid to **negative reviews**, given their business importance and class imbalance.

## 🧰 Tech Stack

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn, XGBoost
- NLTK, VADER, TextBlob
- TensorFlow, Keras
- Hugging Face Transformers (`distilbert-base-uncased`)
- Sentence-BERT (`paraphrase-multilingual-mpnet-base-v2`)
- Google Colab (T4 GPU)

## 📊 Key Insights

- Majority of reviews are overwhelmingly positive (rating 5)
- California park dominates in review volume
- DistilBERT outperformed all models, especially in recall for negative class
- Classical models (SVC, Logistic Regression) were still competitive
- Class imbalance severely impacts model performance—future work should include data augmentation or ensemble methods

