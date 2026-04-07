import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("💳 Credit Risk Prediction System")

# -------- Demographics --------
st.header("👤 Demographics")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

with col3:
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])


# -------- Financial Info --------
st.header("💰 Financial Info")

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input("Income", value=50000)

with col2:
    existing_debts = st.number_input("Existing Debts", value=15000)

with col3:
    savings = st.number_input("Savings", value=7000)

col1, col2, col3 = st.columns(3)

with col1:
    assets = st.selectbox("Assets", ["Car", "Home", "None"])

with col2:
    existing_loans = st.number_input("Existing Loans", value=1)

with col3:
    debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.34)


# -------- Credit History --------
st.header("📊 Credit History")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Credit Score", value=690)

with col2:
    past_defaults = st.number_input("Past Defaults", value=0)

with col3:
    credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.45)

col1, col2, col3 = st.columns(3)

with col1:
    credit_accounts = st.number_input("Number of Credit Accounts", value=3)

with col2:
    credit_inquiries = st.number_input("Credit Inquiries", value=1)

with col3:
    credit_history_length = st.number_input("Credit History Length (Years)", value=5)


# -------- Employment --------
st.header("💼 Employment")

col1, col2, col3 = st.columns(3)

with col1:
    occupation = st.selectbox("Occupation", ["IT", "Finance", "Healthcare", "Education", "Business"])

with col2:
    job_title = st.text_input("Job Title", "Software Engineer")

with col3:
    employment_length = st.number_input("Employment Length (Years)", value=4)

col1, col2 = st.columns(2)

with col1:
    employment_type = st.selectbox("Employment Type", ["Full-Time", "Part-Time", "Self-Employed"])

with col2:
    years_current_job = st.number_input("Years at Current Job", value=3)


# -------- Loan Details --------
st.header("🏦 Loan Details")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input("Loan Amount", value=16000)

with col2:
    loan_term = st.number_input("Loan Term (Months)", value=36)

with col3:
    interest_rate = st.number_input("Interest Rate", value=10)

col1, col2 = st.columns(2)

with col1:
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Business", "Education"])

with col2:
    loan_to_value_ratio = st.slider("Loan-to-Value Ratio", 0.0, 1.0, 0.70)


st.markdown("---")

# -------- Predict Button --------

if st.button("Predict Risk"):

    input_data = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "income": income,
        "existing_debts": existing_debts,
        "savings": savings,
        "assets": assets,
        "existing_loans": existing_loans,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_score": credit_score,
        "past_defaults": past_defaults,
        "credit_utilization": credit_utilization,
        "credit_accounts": credit_accounts,
        "credit_inquiries": credit_inquiries,
        "credit_history_length": credit_history_length,
        "occupation": occupation,
        "job_title": job_title,
        "employment_length": employment_length,
        "employment_type": employment_type,
        "years_current_job": years_current_job,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "interest_rate": interest_rate,
        "loan_purpose": loan_purpose,
        "loan_to_value_ratio": loan_to_value_ratio
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=input_data
        )

        result = response.json()

        st.success("Prediction Successful")

        st.subheader("Risk Score")
        st.write(result["risk_score"])

        st.subheader("Risk Level")
        st.write(result["risk_level"])

        st.subheader("Probabilities")
        st.json(result["probabilities"])

    except:
        st.error("Backend server is not running. Please start FastAPI.")