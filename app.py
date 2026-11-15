"""
Insurance Charges Prediction – Streamlit application

Purpose
-------
Collect customer attributes, apply the same categorical encoding used during
model training, and display the trained regression model's estimated charge.

Generative AI contribution
--------------------------
Generative AI supported the learning and development process by helping explain
Streamlit and pandas concepts, troubleshoot code, and improve documentation.
The prediction itself is produced by the trained scikit-learn model.

Important compatibility note
----------------------------
This app assumes that:
1. insurance_ridge_model.pkl is a fitted model that accepts the encoded columns.
2. feature_columns.pkl contains the exact training feature names and order.
3. Training used one-hot encoding for gender and region, and smoker is encoded
   as 0/1. If the training preprocessing differs, update the app to match it.
"""

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# 1. Page configuration and application introduction
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="💡",
    layout="centered"
)

st.title("Insurance Charges Prediction")
st.write(
    "Estimate medical insurance charges using a trained Machine Learning model. "
    "Enter the customer details below and select Predict."
)


# ---------------------------------------------------------------------------
# 2. Load the trained model and expected feature-column order
# ---------------------------------------------------------------------------
# The model and feature list must be in the same directory as this app.py file,
# or their paths must be updated to point to their actual locations.
@st.cache_resource
def load_model_and_features():
    """Load the fitted model and feature names once per Streamlit session."""
    model = joblib.load("insurance_ridge_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns


try:
    model, feature_columns = load_model_and_features()
except FileNotFoundError as error:
    st.error(
        "A required model file could not be found. Make sure "
        "'insurance_ridge_model.pkl' and 'feature_columns.pkl' are in the "
        "project folder alongside app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the model files: {error}")
    st.stop()


# ---------------------------------------------------------------------------
# 3. Collect customer information
# ---------------------------------------------------------------------------
# Input limits provide basic validation and help prevent unrealistic entries.
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


# ---------------------------------------------------------------------------
# 4. Prepare the input and generate a prediction
# ---------------------------------------------------------------------------
if st.button("Predict Insurance Charges", type="primary"):

    # Build a one-row DataFrame using the same raw feature names as training.
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "no_of_children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region": region
    }])

    # Convert gender and region into one-hot encoded columns.
    # drop_first=True must match the training-time encoding configuration.
    input_encoded = pd.get_dummies(
        input_data,
        columns=["gender", "region"],
        drop_first=True,
        dtype=int
    )

    # Align the input with the exact feature names and order used in training.
    # Missing dummy columns are filled with 0; unexpected columns are removed.
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Generate the estimated charge using the fitted model.
    try:
        prediction = model.predict(input_encoded)[0]
        st.success(f"Estimated Insurance Charges: ${prediction:,.2f}")
        st.caption(
            "This is a model-based estimate for learning and demonstration; "
            "it is not an insurance quote."
        )
    except Exception as error:
        st.error(
            "Prediction failed. Check that the input preprocessing and feature "
            f"columns match the model used during training. Details: {error}"
        )
