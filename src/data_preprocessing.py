import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_and_save_tools(df, save_path='models/'):
    # Target variable 'risk' ko numeric banana (Low:0, Medium:1, High:2)
    risk_mapper = {'Low': 0, 'Medium': 1, 'High': 2}
    if 'risk' in df.columns:
        df['risk'] = df['risk'].map(risk_mapper)

    # Categorical columns identify karna
    categorical_cols = ['gender', 'marital_status', 'assets', 'occupation', 
                        'job_title', 'employment_type', 'loan_purpose']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Encoders save karna taaki API/UI mein use ho sake
    joblib.dump(encoders, f'{save_path}encoders.pkl')
    
    # Features aur Target alag karna
    X = df.drop('risk', axis=1) if 'risk' in df.columns else df
    y = df['risk'] if 'risk' in df.columns else None
    
    # Scaling (Numerical values ko ek range mein lana)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f'{save_path}scaler.pkl')
    
    return X_scaled, y

def load_and_preprocess_input(input_data, save_path='models/'):
    # API ya UI se aane wale single data point ke liye
    encoders = joblib.load(f'{save_path}encoders.pkl')
    scaler = joblib.load(f'{save_path}scaler.pkl')
    
    df = pd.DataFrame([input_data])
    for col, le in encoders.items():
        # Handle unknown categories safely
        df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
    return scaler.transform(df)