import pandas as pd
import os

# Define paths
RAW_PATH = "data/raw/india_housing_prices.csv"
PROCESSED_PATH = "data/processed/final_data.csv"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print("✅ Data loaded successfully")
    return df

def clean_data(df):
    print("🧹 Cleaning data...")

    df = df.drop_duplicates()

    # Fill missing values
    for col in df.select_dtypes(include='number'):
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna(df[col].mode()[0])

    print("✅ Missing values handled")
    return df

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Processed data saved to {path}")

# 🔥 MAIN EXECUTION (THIS WAS MISSING)
if __name__ == "__main__":
    df = load_data(RAW_PATH)
    df = clean_data(df)
    save_data(df, PROCESSED_PATH)