# main.py
from preprocess import load_and_merge, preprocess_data,save_encoders
from model import train_model
from sklearn.model_selection import train_test_split
import joblib
import os
import pandas as pd

def main():
    base_dir = "data/epc"
    cert_path = os.path.join(base_dir, "certificates.csv")
    reco_path = os.path.join(base_dir, "recommendations.csv")
    model_output = os.path.join("models", "epc")

    # Create necessary folders
    os.makedirs(model_output, exist_ok=True)

    df = load_and_merge(cert_path, reco_path)
    X, y, y_encoder, feature_encoders = preprocess_data(df)
    
    # Sample 10% of the data to speed up training
    # X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.9, stratify=y, random_state=42)
    # model = train_model(X_sample, y_sample, output_dir=model_output)
    
    # Train model on the full dataset
    model = train_model(X, y, output_dir=model_output)


    # Save feature columns
    joblib.dump(X.columns.tolist(), "models/epc/feature_columns.pkl")

    # Save encoders
    joblib.dump(y_encoder, os.path.join(model_output, "target_encoder.pkl"))
    save_encoders(feature_encoders, os.path.join(model_output, "feature_encoders.pkl"))
    print("✅ Model and encoders saved successfully to models/epc/")

if __name__ == "__main__":
    main()
