import streamlit as st
import requests
import os
from dotenv import load_dotenv
st.title("Heart Disease Prediction")
API_URL = os.getenv("FASTAPI_URL")
load_dotenv()
# Numeric Inputs
age = st.number_input("Age", 0, 120, 50)
trestbps = st.number_input("Resting BP", 50, 250, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)

# Categorical Inputs
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
fbs = st.selectbox("Fasting BS >120 mg/dl", ["No","Yes"])
restecg = st.selectbox("Resting ECG", ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"])
exang = st.selectbox("Exercise Induced Angina", ["No","Yes"])
slope = st.selectbox("Slope of ST Segment", ["Upsloping","Flat","Downsloping"])
ca = st.selectbox("Major Vessels Colored", ["0","1","2","3"])
thal = st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversable Defect"])

# Mapping for API
sex_map = {"Female":0,"Male":1}
cp_map = {"Typical Angina":0,"Atypical Angina":1,"Non-anginal Pain":2,"Asymptomatic":3}
fbs_map = {"No":0,"Yes":1}
restecg_map = {"Normal":0,"ST-T Abnormality":1,"Left Ventricular Hypertrophy":2}
exang_map = {"No":0,"Yes":1}
slope_map = {"Upsloping":0,"Flat":1,"Downsloping":2}
ca_map = {"0":0,"1":1,"2":2,"3":3}
thal_map = {"Normal":0,"Fixed Defect":1,"Reversable Defect":2}

if st.button("Predict"):
    payload = {
        "age": float(age),
        "sex": sex_map[sex],
        "cp": cp_map[cp],
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": fbs_map[fbs],
        "restecg": restecg_map[restecg],
        "thalach": float(thalach),
        "exang": exang_map[exang],
        "oldpeak": float(oldpeak),
        "slope": slope_map[slope],
        "ca": ca_map[ca],
        "thal": thal_map[thal]
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        res = r.json()
        st.success(f"Prediction: {res['prediction']}")
        st.write(f"Probability (disease): {res['probability']:.2f}")
    except Exception as e:
        st.error(str(e))
