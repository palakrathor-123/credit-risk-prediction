import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from  data_preprocessing import preprocess_and_save_tools

def train_model(csv_path):
    # Data load karein
    df = pd.read_csv(csv_path)
    
    # Preprocessing
    X, y = preprocess_and_save_tools(df)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🚀 Training Random Forest (Baseline)...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = rf_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    # Model save karein
    joblib.dump(rf_model, 'models/final_model.pkl')
    print("✅ Model saved as models/final_model.pkl")

    # Feature Importance (For UI explanation)
    feature_importance = pd.Series(rf_model.feature_importances_, index=pd.read_csv(csv_path).drop('risk', axis=1).columns)
    feature_importance.to_csv('models/feature_importance.csv')

if __name__ == "__main__":
    train_model('data/credit_data.csv')