import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
import io

with open(BASE_DIR / "KNN_heart.pkl", "rb") as f:
    model = joblib.load(io.BytesIO(f.read()))
with open(BASE_DIR / "scaler.pkl", "rb") as f:
    scaler = joblib.load(io.BytesIO(f.read()))
with open(BASE_DIR / "columns.pkl", "rb") as f:
    expected_columns = joblib.load(io.BytesIO(f.read()))

def build_input(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    """Recreate the exact 19-feature contract used during KNN training."""
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": int(sex == "M"),
        "ChestPainType_ATA": int(chest_pain == "ATA"),
        "ChestPainType_NAP": int(chest_pain == "NAP"),
        "ChestPainType_TA": int(chest_pain == "TA"),
        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST": int(resting_ecg == "ST"),
        "ExerciseAngina_Y": int(exercise_angina == "Y"),
        "ST_Slope_Flat": int(st_slope == "Flat"),
        "ST_Slope_Up": int(st_slope == "Up"),
        # Match pandas.cut(..., bins=[0, 40, 55, inf]).
        "AgeGroup_Middle-aged": int(40 < age <= 55),
        "AgeGroup_Senior": int(age > 55),
        # Match pandas.cut(..., bins=[0, 200, 240, inf]).
        "CholesterolLevel_Borderline": int(200 < cholesterol <= 240),
        "CholesterolLevel_High": int(cholesterol > 240),
    }
    input_df = pd.DataFrame([raw_input]).reindex(
        columns=expected_columns, fill_value=0
    )
    if list(input_df.columns) != list(expected_columns):
        raise ValueError("Input feature order does not match columns.pkl")
    if input_df.shape[1] != scaler.n_features_in_:
        raise ValueError("Input feature count does not match scaler.pkl")
    return input_df


st.title("Heart Stroke Prediction By Prem")
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    input_df = build_input(
        age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
        resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
    )
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
