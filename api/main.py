from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(title="Credit Risk Prediction API")


# -------- Input Schema (All Columns) --------

class CreditInput(BaseModel):

    age: int
    gender: str
    marital_status: str
    income: float
    existing_debts: float
    savings: float
    assets: str
    existing_loans: int
    debt_to_income_ratio: float

    credit_score: int
    past_defaults: int
    credit_utilization: float
    credit_accounts: int
    credit_inquiries: int
    credit_history_length: int

    occupation: str
    job_title: str
    employment_length: int
    employment_type: str
    years_current_job: int

    loan_amount: float
    loan_term: int
    interest_rate: float
    loan_purpose: str
    loan_to_value_ratio: float


# -------- Load ML Models --------

model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")


# -------- Home Route --------

@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API is running successfully"}


# -------- Prediction Route --------

@app.post("/predict")
def predict_risk(data: CreditInput):

    # Convert input to dataframe
    df = pd.DataFrame([data.dict()])

    # Encode categorical columns
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )

    # Scale features
    X_scaled = scaler.transform(df)

    # Model Prediction
    prediction = model.predict(X_scaled)[0]

    # Prediction Probability
    probability = model.predict_proba(X_scaled)[0]

    # Risk Level Mapping
    risk_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    risk_level = risk_map[prediction]

    # Risk Score
    risk_score = float(np.max(probability))

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "probabilities": {
            "Low": round(float(probability[0]), 2),
            "Medium": round(float(probability[1]), 2),
            "High": round(float(probability[2]), 2)
        },
        "status": "Success"
    }


# -------- Run Server --------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)