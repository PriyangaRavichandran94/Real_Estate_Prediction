import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import create_features

# =========================
# ✅ Setup
# =========================
print("🚀 Starting training pipeline...")
os.makedirs("models", exist_ok=True)

# =========================
# ✅ Load Data
# =========================
df = pd.read_csv('data/processed/final_data.csv')
print(f"Original Shape: {df.shape}")

# =========================
# ✅ Reduce size (optional)
# =========================
if len(df) > 5000:
    df = df.sample(5000, random_state=42)

# =========================
# ✅ Feature Engineering
# =========================
df = create_features(df)

# 🔥 ENSURE Amenity_Score EXISTS
if "Amenity_Score" not in df.columns:
    print("⚠️ Amenity_Score missing → creating fallback")
    df["Amenity_Score"] = 0

# =========================
# ❗ Drop unwanted
# =========================
df.drop(columns=["ID"], inplace=True, errors="ignore")

# =========================
# ❗ Handle Missing Values (CRITICAL FIX)
# =========================
df = df.fillna({
    col: "Unknown" for col in df.select_dtypes(include="object").columns
})

df = df.fillna(0)

# =========================
# ✅ Targets
# =========================
y_reg = df['Future_Price_5Y']
y_clf = df['Good_Investment']
X = df.drop(['Future_Price_5Y', 'Good_Investment'], axis=1)

# =========================
# ✅ Column Types
# =========================
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()

# Remove high-cardinality
cat_cols = [col for col in cat_cols if X[col].nunique() < 20]

print("Categorical:", cat_cols)
print("Numerical:", num_cols)

# =========================
# ✅ Preprocessing
# =========================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# =========================
# ✅ Pipelines
# =========================
reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    ))
])

clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    ))
])

# =========================
# ✅ SINGLE SPLIT (FIXED)
# =========================
X_train, X_test, y_train_r, y_test_r, y_train_c, y_test_c = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# =========================
# ✅ Train
# =========================
print("🏋️ Training models...")
reg_pipeline.fit(X_train, y_train_r)
clf_pipeline.fit(X_train, y_train_c)

# =========================
# ✅ Predict
# =========================
pred_r = reg_pipeline.predict(X_test)
pred_c = clf_pipeline.predict(X_test)

# =========================
# ✅ Metrics
# =========================
rmse = np.sqrt(mean_squared_error(y_test_r, pred_r))
acc = accuracy_score(y_test_c, pred_c)

print(f"RMSE: {rmse}")
print(f"Accuracy: {acc}")

# =========================
# ✅ MLflow (optional)
# =========================
try:
    mlflow.set_experiment("RealEstate_Final")

    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(reg_pipeline, "regressor")
        mlflow.sklearn.log_model(clf_pipeline, "classifier")

    print("📦 MLflow logged")

except Exception as e:
    print("⚠️ MLflow skipped:", e)

# =========================
# ✅ Save
# =========================
joblib.dump(reg_pipeline, "models/regressor.pkl")
joblib.dump(clf_pipeline, "models/classifier.pkl")

# 🔥 SAVE FEATURE LIST (IMPORTANT FOR APP)
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("💾 Models saved successfully")
print("🎉 DONE")