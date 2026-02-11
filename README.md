
# Machine Learning Classification Assignment – Streamlit App

## Problem Statement
The goal of this project is to implement and compare multiple machine learning classification models on a real dataset. The models are trained and evaluated using standard performance metrics. An interactive Streamlit web app is created to allow users to upload a dataset and test different models.

## Dataset Description
I used the Breast Cancer Classification dataset from sklearn library.
It contains 569 samples and 30 numerical features.
The target variable has two classes representing malignant and benign tumors.

## Models Implemented
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- XGBoost

## Metrics Used
Accuracy, AUC, Precision, Recall, F1 Score, MCC
## Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|---------|------|-----------|---------|------|------|
| Logistic Regression | 0.973684 | 0.997380 | 0.972222 | 0.985915 | 0.979021 | 0.943898 |
| Decision Tree | 0.947368 | 0.943990 | 0.957746 | 0.957746 | 0.957746 | 0.887979 |
| KNN | 0.947368 | 0.981985 | 0.957746 | 0.957746 | 0.957746 | 0.887979 |
| Naive Bayes | 0.964912 | 0.997380 | 0.958904 | 0.985915 | 0.972222 | 0.925285 |
| Random Forest | 0.964912 | 0.995251 | 0.958904 | 0.985915 | 0.972222 | 0.925285 |
| XGBoost | 0.956140 | 0.993122 | 0.958333 | 0.971831 | 0.965035 | 0.906379 |

## Streamlit Features
- CSV upload
- Model selection
- Metrics display
- Confusion matrix
- Classification report
