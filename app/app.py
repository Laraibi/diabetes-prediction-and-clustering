from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Diabetes Risk Predictor")

st.write(
    """
    Enter patient medical information to estimate diabetes risk
    using clustering and supervised machine learning.
    """
)

MODELS_DIR = Path("models")

imputer = joblib.load(MODELS_DIR / "diabetes_imputer.joblib")
classifier = joblib.load(MODELS_DIR / "diabetes_classifier.joblib")
high_risk_cluster = joblib.load(MODELS_DIR / "high_risk_cluster.joblib")

columns_with_missing_values = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

pregnancies = st.slider("Pregnancies", 0, 20, 2)
glucose = st.slider("Glucose", 0, 250, 120)
blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 25)
insulin = st.slider("Insulin", 0, 900, 80)
bmi = st.slider("BMI", 0.0, 70.0, 30.0)
diabetes_pedigree = st.slider(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.5,
)
age = st.slider("Age", 18, 100, 35)

if st.button("Predict Diabetes Risk"):
    input_data = pd.DataFrame([
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age,
        }
    ])

    input_data[columns_with_missing_values] = input_data[
        columns_with_missing_values
    ].replace(0, np.nan)

    input_data[columns_with_missing_values] = imputer.transform(
        input_data[columns_with_missing_values]
    )

    predicted_cluster = classifier.predict(input_data)[0]

    medical_risk = (
        glucose > 126
        and bmi > 30
        and diabetes_pedigree > 0.5
    )

    if predicted_cluster == high_risk_cluster:
        st.error(
            f"High Diabetes Risk — Cluster {predicted_cluster}"
        )

    elif medical_risk:
        st.warning(
            """
            Moderate Risk Detected.

            The clustering model predicts a lower-risk profile,
            but some medical indicators exceed critical thresholds.
            """
        )

    else:
        st.success(
            f"Lower Diabetes Risk — Cluster {predicted_cluster}"
        )

    st.write("Patient input used for prediction:")
    st.dataframe(input_data)