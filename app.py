
import os
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.write("Predict whether a telecom customer is likely to leave the company.")

# Load Trained Model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "churn_model.pkl"
)

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    model_name = artifact.get("model_name", "Trained Model")

except FileNotFoundError:
    st.error(
        "churn_model.pkl nahi mili. Model file ko app.py ke same folder mein rakhein."
    )
    st.stop()

except Exception as e:
    st.error(f"Model load karne mein error: {e}")
    st.stop()

st.caption(f"Model: {model_name}")

# Customer Information Form
st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button("Predict Churn")

# Prediction
if submitted:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        input_df = pd.DataFrame([customer_data])

        # Match training feature columns and order
        input_df = input_df.reindex(columns=feature_columns)

        prediction = int(model.predict(input_df)[0])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Customer is likely to CHURN.")
        else:
            st.success("Customer is likely to STAY.")

        # Show churn probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                churn_probability = probabilities[classes.index(1)]

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2%}"
                )

                st.progress(
                    float(max(0, min(1, churn_probability)))
                )

        with st.expander("View Customer Details"):
            st.dataframe(input_df, use_container_width=True)

        st.caption(
            "This prediction is an estimate, not a guarantee."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
