import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

def evaluate_regressor(model, X_test, y_test):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return rmse

def evaluate_classifier(model, X_test, y_test):
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred)