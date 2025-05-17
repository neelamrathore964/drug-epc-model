import os
import pandas as pd
from preprocess import preprocess, data_summary, eda
from model import train_and_evaluate

def main():
    # Define paths
    base_dir = "Data/Drug_reviews"
    raw_path = os.path.join(base_dir, "drugsComTrain_raw.csv")
    test_path = os.path.join(base_dir, "drugsComTest_raw.csv")
    cleaned_path = os.path.join(base_dir, "cleaned_data")
    eda_path = os.path.join(base_dir, "eda")
    model_output = os.path.join("Models", "Drug_reviews")

    # Create necessary folders
    os.makedirs(cleaned_path, exist_ok=True)
    os.makedirs(eda_path, exist_ok=True)
    os.makedirs(model_output, exist_ok=True)

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(raw_path)
    test_df = pd.read_csv(test_path)

    # Preprocess
    print("Preprocessing...")
    train_clean = preprocess(train_df)
    test_clean = preprocess(test_df)

    #EDA
    print("Exploatory data analysis...")
    eda(train_clean,eda_path)

    # Save cleaned data
    train_clean.to_csv(os.path.join(cleaned_path, "train_clean.csv"), index=False)
    test_clean.to_csv(os.path.join(cleaned_path, "test_clean.csv"), index=False)
    print("Cleaned data saved.")

    # Data Summary
    data_summary(train_clean, test_clean)

    # Train
    print("Training and evaluating models...")
    X_train, y_train = train_clean['clean_review'], train_clean['sentiment_encoded']
    X_test, y_test = test_clean['clean_review'], test_clean['sentiment_encoded']

    train_and_evaluate(X_train, y_train, X_test, y_test, output_dir=model_output)

if __name__ == "__main__":
    main()
