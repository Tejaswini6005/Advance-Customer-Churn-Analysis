import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv('data/customer_churn.csv')
    
    # Data cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Convert Churn to binary
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Select features for model
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner', 'Dependents']
    
    # Convert categorical features
    le = LabelEncoder()
    df['Partner'] = le.fit_transform(df['Partner'])
    df['Dependents'] = le.fit_transform(df['Dependents'])
    
    X = df[features]
    y = df['Churn']
    
    return X, y, df

def train_model():
    X, y, df = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(model, 'churn_model.pkl')
    print("Model saved as churn_model.pkl")
    
    return model, X_test, y_test

if __name__ == "__main__":
    train_model()