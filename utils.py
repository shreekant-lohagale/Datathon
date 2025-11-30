import pandas as pd
import numpy as np
import joblib

def clean_dataframe(df):
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna("Unknown")
    return df

def engineer_features(df):
    df = df.copy()
    # Example feature engineering (customize based on your model)
    df["log_sqft"] = np.log1p(df.get("sqft", 1))
    return df

def load_model():
    return joblib.load("models/model.pkl")
