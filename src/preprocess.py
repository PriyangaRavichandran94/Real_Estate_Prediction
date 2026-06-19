# src/data_preprocessing.py

import pandas as pd
import os

# -----------------------------
# PATHS
# -----------------------------
RAW_PATH = "data/raw/india_housing_prices.csv"
PROCESSED_PATH = "data/processed/final_data.csv"


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")

    df = pd.read_csv(path)
    print("✅ Data loaded successfully")
    print(f"Shape: {df.shape}")
    return df


# -----------------------------
# CLEAN DATA
# -----------------------------
def clean_data(df):
    print("🧹 Cleaning data...")

    # Remove duplicates
    df = df.drop_duplicates()

    print(f"✅ Duplicates removed. Shape: {df.shape}")
    return df


# -----------------------------
# PREPROCESS DATA
# -----------------------------
def preprocess_data(df):
    print("⚙️ Preprocessing data...")

    # -------------------------------------------------
    # Missing Value Handling
    # -------------------------------------------------

    # Parking spaces → 0 if missing
    if "Parking_Space" in df.columns:
        df["Parking_Space"] = df["Parking_Space"].fillna(0)

    # Nearby facilities → median
    for col in ["Nearby_Schools", "Nearby_Hospitals"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Numeric columns → median
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns → mode
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # -------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------

    # Price per SqFt
    if (
        "Price_in_Lakhs" in df.columns
        and "Size_in_SqFt" in df.columns
    ):
        df["Price_per_SqFt"] = (
            df["Price_in_Lakhs"] * 100000
        ) / df["Size_in_SqFt"]

    # Age of Property
    if "Year_Built" in df.columns:
        CURRENT_YEAR = 2025
        df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]

    # -------------------------------------------------
    # Keep Only Required Dataset Columns
    # -------------------------------------------------

    required_columns = [
        "ID",
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Furnished_Status",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Amenities",
        "Facing",
        "Owner_Type",
        "Availability_Status"
    ]

    # Keep only columns that exist
    available_columns = [
        col for col in required_columns
        if col in df.columns
    ]

    df = df[available_columns]

    print("✅ Feature engineering completed")
    print(f"Final Shape: {df.shape}")

    return df


# -----------------------------
# SAVE DATA
# -----------------------------
def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print(f"✅ Processed data saved to: {path}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    try:
        df = load_data(RAW_PATH)

        df = clean_data(df)

        df = preprocess_data(df)

        save_data(df, PROCESSED_PATH)

        print("\n🎉 Data preprocessing completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")