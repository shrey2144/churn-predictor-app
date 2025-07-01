import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model("churn_model.h5")

st.title("Customer Churn Prediction App")

uploaded_file = st.file_uploader("Upload Customer CSV File", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Original Data Preview:", df.head())

    try:
        df.drop('customerID', axis=1, inplace=True)
        df = df[df.TotalCharges != ' ']
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.replace('No internet service', 'No', inplace=True)
        df.replace('No phone service', 'No', inplace=True)

        yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                          'StreamingMovies', 'PaperlessBilling']
        for col in yes_no_columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0})
        df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})

        df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])

        # Ensure all expected columns from training are present
        expected_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
            'MonthlyCharges', 'TotalCharges',
            'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
            'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
            'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
        ]

        # Add any missing dummy columns as 0
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder the columns to match model input
        df = df[expected_columns]

        # Scale numerical columns
        scaler = MinMaxScaler()
        cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        expected_cols = model.input_shape[1]
        if df.shape[1] != expected_cols:
            st.error(f"Model expected {expected_cols} features but got {df.shape[1]}. Check column encoding.")
        else:
            predictions = model.predict(df)
            df['Churn Probability'] = predictions
            df['Predicted Churn'] = (df['Churn Probability'] > 0.5).astype(int)

            st.subheader("Prediction Results")
            st.write(df[['Churn Probability', 'Predicted Churn']].head())

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name='churn_predictions.csv', mime='text/csv')

    except Exception as e:
        st.error(f"Error processing file: {e}")
