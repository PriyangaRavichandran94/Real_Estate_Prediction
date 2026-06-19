import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBClassifier, XGBRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("data/processed/final_data.csv")

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

city_growth = {
    "Chennai":0.11,
    "Bangalore":0.14,
    "Hyderabad":0.13,
    "Mumbai":0.10,
    "Delhi":0.09,
    "Pune":0.12
}

df["Growth_Rate"] = df["City"].map(city_growth).fillna(0.08)

df["Amenity_Density_Score"] = (
    df["Amenities"].astype(str).str.count(",") + 1
)

df["Future_Price_5Y"] = (
    df["Price_in_Lakhs"]
    * ((1 + df["Growth_Rate"]) ** 5)
)

median_ppsft = df["Price_per_SqFt"].median()

df["Good_Investment"] = (
    (df["Growth_Rate"] >= 0.10) &
    (df["Price_per_SqFt"] < median_ppsft) &
    (df["Age_of_Property"] <= 15)
).astype(int)

transport_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

df["Public_Transport_Accessibility"] = (
    df["Public_Transport_Accessibility"]
    .map(transport_map)
)


# Parking Space
try:
    df["Parking_Space"] = pd.to_numeric(
        df["Parking_Space"],
        errors="raise"
    )
except:
    parking_map = {
        "No": 0,
        "Yes": 1
    }

    df["Parking_Space"] = (
        df["Parking_Space"]
        .map(parking_map)
    )

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------

encoders = {}

cat_cols = [
    "State",
    "City",
    "Property_Type",
    "Furnished_Status",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status"
]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, "models/label_encoders.pkl")

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------

drop_cols = [
    "ID",
    "Good_Investment",
    "Future_Price_5Y"
    
]

df.drop(columns=["Locality"], inplace=True)

X = df.drop(columns=drop_cols)


joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)

y_class = df["Good_Investment"]
y_reg = df["Future_Price_5Y"]

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------

scaler = StandardScaler()
print("\nRemaining object columns:")
print(
    df.select_dtypes(include="object")
      .columns
      .tolist()
)

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# ---------------------------------------------------
# SPLIT
# ---------------------------------------------------

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled,
    y_class,
    test_size=0.2,
    random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_scaled,
    y_reg,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# CLASSIFICATION MODELS
# ---------------------------------------------------

classifiers = {
    "LogisticRegression":
        LogisticRegression(max_iter=1000),

    "RandomForest":
        RandomForestClassifier(),

    "GradientBoosting":
        GradientBoostingClassifier(),

    "ExtraTrees":
        ExtraTreesClassifier(),

    "XGBoost":
        XGBClassifier()
}

best_auc = 0

for name, model in classifiers.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train_c, y_train_c)

        preds = model.predict(X_test_c)

        probs = model.predict_proba(X_test_c)[:,1]

        acc = accuracy_score(y_test_c,preds)
        prec = precision_score(y_test_c,preds)
        rec = recall_score(y_test_c,preds)
        auc = roc_auc_score(y_test_c,probs)

        mlflow.log_param("model",name)
        mlflow.log_metric("accuracy",acc)
        mlflow.log_metric("precision",prec)
        mlflow.log_metric("recall",rec)
        mlflow.log_metric("roc_auc",auc)

        mlflow.sklearn.log_model(model,name)

        if auc > best_auc:
            best_auc = auc
            best_classifier = model

joblib.dump(
    best_classifier,
    "models/best_classifier.pkl"
)

# ---------------------------------------------------
# REGRESSION MODELS
# ---------------------------------------------------

from sklearn.linear_model import LinearRegression

regressors = {

    "LinearRegression":
        LinearRegression(),

    "RandomForest":
        RandomForestRegressor(),

    "GradientBoosting":
        GradientBoostingRegressor(),

    "ExtraTrees":
        ExtraTreesRegressor(),

    "XGBoost":
        XGBRegressor()
}

best_r2 = -999

for name, model in regressors.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train_r, y_train_r)

        preds = model.predict(X_test_r)

        rmse = np.sqrt(
            mean_squared_error(y_test_r,preds)
        )

        mae = mean_absolute_error(
            y_test_r,preds
        )

        r2 = r2_score(
            y_test_r,preds
        )

        mlflow.log_param("model",name)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)

        mlflow.sklearn.log_model(model,name)

        if r2 > best_r2:
            best_r2 = r2
            best_regressor = model

joblib.dump(
    best_regressor,
    "models/best_regressor.pkl"
)

print("Training completed")