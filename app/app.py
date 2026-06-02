# src/app.py

import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# ✅ LOAD MODELS
# -----------------------------
regressor = joblib.load("models/regressor.pkl")
classifier = joblib.load("models/classifier.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Load dataset (for dropdown values)
df = pd.read_csv("data/processed/final_data.csv")

st.set_page_config(page_title="Real Estate Predictor", layout="wide")
st.title("🏠 Real Estate Price & Investment Prediction")

# -----------------------------
# ✅ SAFE INPUT FUNCTIONS
# -----------------------------
def safe_selectbox(label, column):
    if column in df.columns:
        values = sorted(df[column].dropna().astype(str).unique())
        return st.selectbox(label, values)
    return "Unknown"

def safe_number(label, default):
    return st.number_input(label, value=default)

# -----------------------------
# 🎯 USER INPUTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    city = safe_selectbox("City", "City")
    locality = safe_selectbox("Locality", "Locality")
    property_type = safe_selectbox("Property Type", "Property_Type")

    bhk = st.slider("BHK", 1, 5, 2)
    size = safe_number("Size (SqFt)", 1000)

    furnished = safe_selectbox("Furnished Status", "Furnished_Status")
    owner_type = safe_selectbox("Owner Type", "Owner_Type")

with col2:
    floor_no = safe_number("Floor Number", 2)
    total_floors = safe_number("Total Floors", 5)

    parking = safe_selectbox("Parking", "Parking_Space")
    security = safe_selectbox("Security", "Security")

    facing = safe_selectbox("Facing", "Facing")
    availability = safe_selectbox("Availability", "Availability_Status")

    # 🔥 IMPORTANT: Keep SAME TYPE as dataset
    schools = safe_selectbox("Nearby Schools", "Nearby_Schools")
    hospitals = safe_selectbox("Nearby Hospitals", "Nearby_Hospitals")
    transport = safe_selectbox("Public Transport", "Public_Transport_Accessibility")

# -----------------------------
# 🧩 AMENITIES (MULTISELECT)
# -----------------------------
amenities_options = ["Gym", "Pool", "Lift", "Garden", "Clubhouse"]
selected_amenities = st.multiselect("Amenities", amenities_options)

amenity_score = len(selected_amenities)

# -----------------------------
# 🎯 INPUT DICTIONARY
# -----------------------------
input_dict = {
    "City": city,
    "Locality": locality,
    "Property_Type": property_type,
    "BHK": bhk,
    "Size_in_SqFt": size,
    "Furnished_Status": furnished,
    "Owner_Type": owner_type,
    "Floor_No": floor_no,
    "Total_Floors": total_floors,
    "Parking_Space": parking,
    "Security": security,
    "Facing": facing,
    "Availability_Status": availability,
    "Nearby_Schools": schools,
    "Nearby_Hospitals": hospitals,
    "Public_Transport_Accessibility": transport,
    "Amenity_Score": amenity_score
}

# -----------------------------
# ✅ CREATE DATAFRAME
# -----------------------------
input_df = pd.DataFrame([input_dict])

# -----------------------------
# 🔥 HANDLE MISSING VALUES BEFORE ALIGNMENT
# -----------------------------
input_df = input_df.fillna("Unknown")

# -----------------------------
# 🔥 MATCH TRAINING COLUMNS
# -----------------------------
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# 🎯 PREDICTION
# -----------------------------
if st.button("🔮 Predict"):
    try:
        price = regressor.predict(input_df)[0]
        invest = classifier.predict(input_df)[0]

        st.success(f"💰 Estimated Future Price: ₹ {price:.2f} Lakhs")

        if invest == 1:
            st.success("✅ Good Investment Opportunity")
        else:
            st.warning("⚠️ Risky Investment")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")