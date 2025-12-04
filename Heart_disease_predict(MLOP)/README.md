
# Heart Disease Prediction 

This is a **Heart Disease Prediction web application** built using **FastAPI** (backend API) and **Streamlit** (frontend UI).  
It predicts the likelihood of heart disease based on patient clinical features.

---

## Features

- Predict heart disease using **clinical inputs**:
  - Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, ST Depression, Slope, Number of Vessels, Thalassemia
- **Human-readable predictions**: "Heart Disease" / "No Heart Disease"
- **Probability/confidence score** shown for model prediction
- **Interactive Streamlit UI** for easy data input
- **FastAPI backend** for serving model predictions
- Uses **pre-trained RandomForest model** with scaler

---



---

## Setup & Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd heart-disease-demo
````

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure **model.pkl** and **scaler.pkl** are present in the project folder.

---

## Running the Application

### 1. Start FastAPI Backend

```bash
uvicorn app:app --reload
```

* The API runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger Docs available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Start Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

* Open browser 
* Enter patient details and click **Predict**

---

## Sample High-Risk Patient Input (Heart Disease Likely)

| Feature         | Value          |
| --------------- | -------------- |
| Age             | 60             |
| Sex             | Male           |
| Chest Pain Type | Typical Angina |
| Resting BP      | 150            |
| Cholesterol     | 260            |
| Fasting BS >120 | Yes            |
| Resting ECG     | Abnormal       |
| Max Heart Rate  | 120            |
| Exercise Angina | Yes            |
| ST Depression   | 2.5            |
| Slope           | Downsloping    |
| Major Vessels   | 2              |
| Thalassemia     | Defect         |

> Input these values in Streamlit to see a **“Heart Disease” prediction**.

---

## API Endpoint

### POST `/predict`

* **Request JSON**:

```json
{
  "age": 60,
  "sex": 1,
  "cp": 0,
  "trestbps": 150,
  "chol": 260,
  "fbs": 1,
  "restecg": 1,
  "thalach": 120,
  "exang": 1,
  "oldpeak": 2.5,
  "slope": 2,
  "ca": 2,
  "thal": 2
}
```

* **Response JSON**:

```json
{
  "prediction": "Heart Disease",
  "probability": 0.85,
  "detail": "success"
}
```

---

## Notes

* The model is **for demonstration purposes only** and **not a medical diagnosis tool**.
* Make sure **scaler.pkl** is consistent with **model.pkl**, otherwise predictions will be inaccurate.
* Inputs are validated via **dropdowns and number fields** to prevent invalid entries.

---

## Dependencies

* Python 3.10+
* fastapi
* uvicorn
* pydantic
* numpy
* joblib
* scikit-learn
* streamlit
* requests

---

## Docker Deployment

You can containerize the app using Docker:

1. Build the image:

```bash
docker build -t heart-disease-demo .
```

2. Run the container:

```bash
docker run -p 8501:8501 -p 8000:8000 heart-disease-demo
```


---

## Author

Asad Murtaza Shah

