import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Trains a Linear Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        
    Returns:
        LinearRegression: Trained model object.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model successfully trained.")
    return model

def evaluate_model(
    model: LinearRegression, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Predicts values and computes evaluation metrics (MAE, RMSE, R2).
    
    Args:
        model (LinearRegression): Trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        
    Returns:
        Tuple[np.ndarray, Dict[str, float]]: Predictions array and a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    return y_pred, metrics

def get_feature_coefficients(model: LinearRegression, features: List[str]) -> pd.DataFrame:
    """
    Creates a DataFrame mapping features to their linear coefficients.
    
    Args:
        model (LinearRegression): Trained model.
        features (List[str]): List of feature names.
        
    Returns:
        pd.DataFrame: Table of features and their coefficients, sorted by absolute coefficient magnitude descending.
    """
    coefs_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    })
    # Add absolute coefficient magnitude for descriptive sorting
    coefs_df["Abs_Coefficient"] = coefs_df["Coefficient"].abs()
    coefs_df = coefs_df.sort_values(by="Abs_Coefficient", ascending=False).drop(columns="Abs_Coefficient")
    return coefs_df.reset_index(drop=True)
