import joblib
import pandas as pd

# Load saved files
model = joblib.load("models/heart_model.pkl")   # or KNN_heart.pkl if you kept that name
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")


def predict_heart_disease(input_data):
    """
    Predict heart disease from user input.

    Returns:
        prediction (int): 0 or 1
        confidence (float): Prediction confidence (%)
    """

    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    # These checks are mainly for robustness.
    # Your Streamlit sliders already prevent users from entering 0.

    if df.loc[0, "RestingBP"] == 0:
        df.loc[0, "RestingBP"] = resting_bp_mean # Mean used during training

    if df.loc[0, "Cholesterol"] == 0:
        df.loc[0, "Cholesterol"] = cholesterol_mean  # Mean used during training

    # -----------------------------
    # One-Hot Encoding
    # -----------------------------
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # -----------------------------
    # Feature Scaling
    # -----------------------------
    df_scaled = scaler.transform(df)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(df_scaled)[0]

    # Probability (KNN supports predict_proba)
    probability = model.predict_proba(df_scaled)[0]

    confidence = max(probability) * 100

    return prediction, confidence