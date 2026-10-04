import sys
from pathlib import Path

# Find the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
import streamlit as st

from config import MODEL_DIR, FEATURES


# -----------------------------
# Streamlit page configuration
# -----------------------------

st.set_page_config(
    page_title="Heart Disease Diagnostic Aid",
    page_icon="❤️",
    layout="centered",
)

st.title("Heart Disease Diagnostic Aid")

st.caption(
    "Research / educational prototype — not a medical diagnosis."
)


# -----------------------------
# Load trained model
# -----------------------------

MODEL_PATH = MODEL_DIR / "random_forest.joblib"

if not MODEL_PATH.exists():
    st.error(
        "Model not found. Please run `python src/train.py` first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)


# -----------------------------
# Patient information
# -----------------------------

st.subheader("Patient information")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=52
    )

    sex = st.selectbox(
        "Sex (dataset code)",
        [0, 1],
        index=1
    )

    cp = st.selectbox(
        "Chest pain type (cp)",
        [0, 1, 2, 3],
        index=0
    )

    trestbps = st.number_input(
        "Resting blood pressure",
        min_value=50,
        max_value=250,
        value=130
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=220
    )

    fbs = st.selectbox(
        "Fasting blood sugar > 120 (fbs)",
        [0, 1],
        index=0
    )

    restecg = st.selectbox(
        "Resting ECG (restecg)",
        [0, 1, 2],
        index=1
    )


with col2:

    thalach = st.number_input(
        "Maximum heart rate (thalach)",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise-induced angina (exang)",
        [0, 1],
        index=0
    )

    oldpeak = st.number_input(
        "ST depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope",
        [0, 1, 2],
        index=1
    )

    ca = st.selectbox(
        "Number of major vessels (ca)",
        [0, 1, 2, 3, 4],
        index=0
    )

    thal = st.selectbox(
        "Thalassemia code (thal)",
        [0, 1, 2, 3],
        index=2)


# -----------------------------
# Create input dataframe
# -----------------------------

row = pd.DataFrame([{

    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,

}])[FEATURES]


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict risk", type="primary"):

    probability = float(
        model.predict_proba(row)[0, 1]
    )

    prediction = int(
        model.predict(row)[0]
    )

    st.metric(
        "Predicted probability",
        f"{probability:.1%}"
    )

    if prediction == 1:

        st.warning(
            "The model predicts the positive class."
        )

    else:

        st.success(
            "The model predicts the negative class."
        )

    st.info(
        "This output is a machine-learning prediction "
        "from the supplied research dataset. "
        "It must not be interpreted as a clinical diagnosis."
    )
