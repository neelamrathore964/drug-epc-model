# preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_merge(cert_path, reco_path):
    cert_df = pd.read_csv(cert_path, low_memory=False)
    reco_df = pd.read_csv(reco_path, low_memory=False)

    # Merge on LMK_KEY or BUILDING_REFERENCE_NUMBER — adjust if needed
    if "LMK_KEY" in cert_df.columns and "LMK_KEY" in reco_df.columns:
        df = pd.merge(cert_df, reco_df, on="LMK_KEY", how="left")
    elif "BUILDING_REFERENCE_NUMBER" in cert_df.columns and "BUILDING_REFERENCE_NUMBER" in reco_df.columns:
        df = pd.merge(cert_df, reco_df, on="BUILDING_REFERENCE_NUMBER", how="left")
    else:
        raise ValueError("No common key to merge the files.")

    return df

def preprocess_data(df):
    
    # Drop high-cardinality or identifier columns and leakage
    drop_cols = [
        "LMK_KEY", "ADDRESS1", "ADDRESS2", "ADDRESS3", "POSTCODE",
        "BUILDING_REFERENCE_NUMBER", "INSPECTION_DATE", "LODGERMENT_DATE",
        "LODGERMENT_DATETIME", "ADDRESS", "UPRN", "UPRN_SOURCE",
        "CURRENT_ENERGY_RATING", "CURRENT_ENERGY_EFFICIENCY", "POTENTIAL_ENERGY_EFFICIENCY"
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # Target variable
    y = df["POTENTIAL_ENERGY_RATING"]
    df.drop(columns=["POTENTIAL_ENERGY_RATING"], inplace=True)

    # Encode target labels
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)

    # Drop columns with >40% missing
    df = df.loc[:, df.isnull().mean() < 0.4]

    # Separate numeric and categorical
    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns

    # Fill missing
    for col in cat_cols:
        df.loc[:, col] = df[col].fillna("missing")

    for col in num_cols:
        df.loc[:, col] = df[col].fillna(df[col].median())


    # Encode categorical features
    le_dict = {}
    for col in cat_cols:
        df.loc[:, col] = df[col].astype(str)
        le = LabelEncoder()
        df.loc[:, col] = le.fit_transform(df[col])
        le_dict[col] = le  # Save encoder

    return df, y, y_le, le_dict

def save_encoders(encoder_dict, path="models/epc/feature_encoders.pkl"):
    joblib.dump(encoder_dict, path)
    print(f"✅ Feature encoders saved to {path}")

def load_encoders(path="models/epc/feature_encoders.pkl"):
    return joblib.load(path)