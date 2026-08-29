import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("student_performance_model.pkl")

# Load dataset
df = pd.read_csv("data/student-por.csv", sep=";")

# Clean column names
df.columns = df.columns.str.strip()

st.title("Student Performance Prediction")
st.write("Predict student final G3 score using Machine Learning")

# Check columns
st.write("Dataset Columns:", df.columns.tolist())

# Remove G3 only if it exists
if "G3" in df.columns:
    X = df.drop(columns=["G3"])
else:
    X = df.copy()

# Encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Match training columns
X_encoded = X_encoded.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

st.write("Encoded Features:", X_encoded.shape)

# Student selection
student_index = st.number_input(
    "Enter Student Row Number",
    min_value=0,
    max_value=len(X_encoded) - 1,
    value=0,
    step=1
)

# Prediction
if st.button("Predict Performance"):

    student_data = X_encoded.iloc[[student_index]]

    prediction = model.predict(student_data)

    st.success(
        f"Predicted Final G3 Score: {prediction[0]:.2f}"
    )