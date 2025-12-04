from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import traceback


# Pydantic
class PredictRequest(BaseModel):
    age: float = Field(..., description="Age in years")
    sex: int = Field(..., description="0=Female, 1=Male")
    cp: int = Field(..., description="Chest pain type (0-3)")
    trestbps: float = Field(..., description="Resting BP")
    chol: float = Field(..., description="Cholesterol")
    fbs: int = Field(..., description="Fasting blood sugar >120 mg/dl")
    restecg: int = Field(..., description="Resting ECG results")
    thalach: float = Field(..., description="Max heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., description="Slope of peak exercise ST segment")
    ca: int = Field(..., description="Number of major vessels colored")
    thal: int = Field(..., description="Thalassemia (0=normal,1=fixed,2=reversable)")

class PredictResponse(BaseModel):
    prediction: str
    probability: float = None
    detail: str = "success"

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found!")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

feature_order = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

def preprocess(input_data: dict):
    arr = np.array([input_data[name] for name in feature_order], dtype=float).reshape(1, -1)
    return scaler.transform(arr)

def predict(input_data: dict):
    x_scaled = preprocess(input_data)
    pred = int(model.predict(x_scaled)[0])
    proba = float(model.predict_proba(x_scaled)[0,1]) if hasattr(model, "predict_proba") else None
    label_map = {0:"No Heart Disease", 1:"Heart Disease"}
    return label_map[pred], proba


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease from clinical features",
    version="1.0"
)

@app.get("/")
def root():
    return {"status":"ok", "message":"API is running."}

@app.post("/predict", response_model=PredictResponse)
def make_prediction(req: PredictRequest):
    try:
        pred_label, proba = predict(req.dict())
        return PredictResponse(prediction=pred_label, probability=proba)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
