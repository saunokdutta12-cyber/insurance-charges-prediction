
import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature names
model = joblib.load("insurance_ridge_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Insurance Charges Prediction")
st.write("Predict medical insurance charges using Machine Learning.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox("Gender", ["female", "male"])

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox("Smoker", ["no", "yes"])

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Prediction button
if st.button("Predict Insurance Charges"):

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "no_of_children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region": region
    }])

    # Apply the same encoding used during training
    input_encoded = pd.get_dummies(
        input_data,
        columns=["gender", "region"],
        drop_first=True,
        dtype=int
    )

    # Match the training feature columns and order
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict
    prediction = model.predict(input_encoded)[0]

    st.success(
        f"Estimated Insurance Charges: ${prediction:,.2f}"
    )
