# Credit-Risk-Prediction
A Machine Learning based web application that predicts the credit risk of a customer based on demographic, financial, credit history, employment, and loan details.

The system uses a trained machine learning model to calculate the *Risk Score* and classify the customer into *Low, Medium, or High Risk*.

---

## Project Overview

Banks and financial institutions need to evaluate the risk before approving loans. This project helps in predicting whether a customer is likely to default on a loan.

This application takes customer information as input and returns:

- Risk Score
- Risk Level (Low / Medium / High)
- Probability of each risk category

---

## Features

- Machine Learning based prediction
- FastAPI backend for model serving
- Streamlit UI for interactive user interface
- Risk Score prediction
- Probability distribution for risk classes
- Organized input sections for better usability

---

## Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Streamlit

### Tools
- VS Code
- Git
- GitHub

---

## Project Structure
credit-risk-prediction │ ├── api │   └── main.py │ ├── ui │   └── app.py │ ├── models │   ├── final_model.pkl │   ├── scaler.pkl │   └── encoders.pkl │ ├── data │   └── credit_data.csv │ ├── src │   ├── data_preprocessing.py │   ├── model_trainer.py │   └── utils.py │ ├── requirements.txt └── README.md
---

## Input Features

The system uses the following categories of input data:

### Demographics
- Age
- Gender
- Marital Status

### Financial Information
- Income
- Existing Debts
- Savings
- Assets
- Existing Loans
- Debt-to-Income Ratio

### Credit History
- Credit Score
- Past Defaults
- Credit Utilization
- Number of Credit Accounts
- Credit Inquiries
- Credit History Length

### Employment
- Occupation
- Job Title
- Employment Length
- Employment Type
- Years at Current Job

### Loan Details
- Loan Amount
- Loan Term
- Interest Rate
- Loan Purpose
- Loan-to-Value Ratio

---

## Output

The system provides the following predictions:

- *Risk Score* (Probability of Default)
- *Risk Level*
  - Low Risk
  - Medium Risk
  - High Risk
- *Probability Distribution*
  - Probability of Low Risk
  - Probability of Medium Risk
  - Probability of High Risk

---

## Installation

Clone the repository:
git clone https://github.com/your-username/credit-risk-prediction.git


Move to the project directory:
cd credit-risk-prediction


Install dependencies:
pip install -r requirements.txt


---

## Running the Application

### Start FastAPI Backend
uvicorn api.main:app --reload


Backend will run at:
http://127.0.0.1:8000


---

### Start Streamlit UI
streamlit run ui/app.py


UI will open at:
http://localhost:8501


---
## Dashboard Overview

