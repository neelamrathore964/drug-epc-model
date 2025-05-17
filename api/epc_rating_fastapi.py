from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from src.epc.preprocess import preprocess_data,load_and_merge

# Initialize FastAPI app
app = FastAPI(title="🏠 EPC Rating Predictor API")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")  # Store this in your .env file

# Model paths
model_path = "models/epc/best_model.pkl"
encoder_path = "models/epc/target_encoder.pkl"
cert_path = "data/epc/certificates.csv"
reco_path = "data/epc/recommendations.csv"
feature_encoders = joblib.load("models/epc/feature_encoders.pkl")
feature_columns = joblib.load("models/epc/feature_columns.pkl")

# Load model and encoder with checkpoints
try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model from {model_path}: {e}")

try:
    target_encoder = joblib.load(encoder_path)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load encoder from {encoder_path}: {e}")

# Define request schema
class EPCRequest(BaseModel):
    lmk_key: str = None
    address: str = None

# Load property data from file
def load_property_features(lmk_key: str = None, address: str = None):
    try:
        df = load_and_merge(cert_path, reco_path)  # Use merged data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to load and merge data: {e}")

    try:
        if lmk_key:
            prop = df[df["LMK_KEY"] == lmk_key]
        elif address:
            prop = df[df["ADDRESS1"].str.contains(address, case=False, na=False)]
        else:
            raise HTTPException(status_code=400, detail="❌ Please provide either LMK_KEY or address.")

        if prop.empty:
            raise HTTPException(status_code=404, detail="❌ Property not found.")

        return prop.iloc[0:1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed during property lookup: {e}")

# Preprocess the raw data
def preprocess_property(df_raw: pd.DataFrame):
    try:
        df = df_raw.copy()
        df["POTENTIAL_ENERGY_RATING"] = "C"  # Dummy for preprocessing
        features, _, _, _ = preprocess_data(df)  # Get full return

        # Align features to match training columns
        for col in feature_columns:
            if col not in features.columns:
                features[col] = 0  # or 'missing' or np.nan depending on type

        features = features[feature_columns]  # Reorder and drop extras
        return features
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"❌ Preprocessing ValueError: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Unexpected preprocessing error: {e}")

# Prediction endpoint
@app.post("/predict_epc_rating")
def predict_epc_rating(data: EPCRequest, x_api_key: str = Header(...)):
    # Step 1: API Key Validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="❌ Invalid API Key")

    try:
        # Step 2: Load data from CSV
        raw_data = load_property_features(lmk_key=data.lmk_key, address=data.address)

        # Step 3: Preprocess the data
        features = preprocess_property(raw_data)

        # Step 4: Make prediction
        prediction = model.predict(features)[0]

        # Step 5: Decode label
        rating = target_encoder.inverse_transform([prediction])[0]

        return {
            "lmk_key": data.lmk_key,
            "address": data.address,
            "predicted_potential_energy_rating": rating,
        }

    except HTTPException:
        raise  # Let FastAPI return expected HTTPException status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ General prediction error: {e}")
 