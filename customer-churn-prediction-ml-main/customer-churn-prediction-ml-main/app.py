
import streamlit as st
import joblib
import numpy as np

st.title("Customer Churn Prediction App")

# Load trained model
model = joblib.load("churn_model.pkl")

st.write("Enter Customer Details")

tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

if st.button("Predict Churn"):
    features = np.array([[tenure, monthly_charges, total_charges]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is NOT likely to Churn")

