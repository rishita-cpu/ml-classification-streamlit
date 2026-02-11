import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report


st.title("ML Classification Model Demo App")

st.write("Upload a CSV test dataset and select a model to evaluate")

# -------------------------
# Load saved models
# -------------------------

lr_model = joblib.load("logistic_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
knn_model = joblib.load("knn_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

scaler = joblib.load("scaler.pkl")

# -------------------------
# Model selection
# -------------------------

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -------------------------
# File upload
# -------------------------

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(data.head())

    # Assume last column is target (simple beginner assumption)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # scale features (for models that need it)
    X_scaled = scaler.transform(X)

    # -------------------------
    # Select model object
    # -------------------------

    if model_name == "Logistic Regression":
        model = lr_model
        X_use = X_scaled

    elif model_name == "Decision Tree":
        model = dt_model
        X_use = X

    elif model_name == "KNN":
        model = knn_model
        X_use = X_scaled

    elif model_name == "Naive Bayes":
        model = nb_model
        X_use = X_scaled

    elif model_name == "Random Forest":
        model = rf_model
        X_use = X

    else:
        model = xgb_model
        X_use = X

    # -------------------------
    # Prediction
    # -------------------------

    y_pred = model.predict(X_use)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_use)[:,1]
    else:
        y_prob = y_pred

    # -------------------------
    # Metrics
    # -------------------------

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)
    st.write("AUC:", auc)
    st.write("MCC:", mcc)

    # -------------------------
    # Confusion Matrix
    # -------------------------

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))