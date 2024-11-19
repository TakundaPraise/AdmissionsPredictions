import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load Pre-trained Models and Encoders
@st.cache
def load_models():
    admission_model = pickle.load(open("admission_model.pkl", "rb"))
    program_model = pickle.load(open("program_model.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return admission_model, program_model, label_encoder, scaler

# Load models
admission_model, program_model, label_encoder, scaler = load_models()

# Streamlit UI
st.title("University Admission Prediction System")

# Input Section
st.header("Enter Student Details")
applicant_id = st.text_input("Applicant ID", "")
gender = st.selectbox("Gender", ["Male", "Female"])
o_level_maths = st.selectbox("O-Level Maths Grade", ["A", "B", "C", "D", "E", "F"])
o_level_english = st.selectbox("O-Level English Grade", ["A", "B", "C", "D", "E", "F"])
o_level_science = st.selectbox("O-Level Science Grade", ["A", "B", "C", "D", "E", "F"])
a_level_grades = {
    "Grade 1": st.selectbox("A-Level Grade 1", ["A", "B", "C", "D", "E"]),
    "Grade 2": st.selectbox("A-Level Grade 2", ["A", "B", "C", "D", "E"]),
    "Grade 3": st.selectbox("A-Level Grade 3", ["A", "B", "C", "D", "E"]),
}
program_choices = [
    st.selectbox("Program 1 Choice", ["BSc Nursing Education", "BSc Engineering", "BCom Accounting"]),
    st.selectbox("Program 2 Choice", ["BSc Nursing Education", "BSc Engineering", "BCom Accounting"]),
    st.selectbox("Program 3 Choice", ["BSc Nursing Education", "BSc Engineering", "BCom Accounting"]),
]

# Mapping Grades to Numbers
def o_level_pass_fail(grade):
    return 1 if grade in ["A", "B", "C"] else 0

grade_to_points = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 0}

# Process Input
input_data = {
    "Gender": [gender],
    "O-Level Maths": [o_level_pass_fail(o_level_maths)],
    "O-Level English": [o_level_pass_fail(o_level_english)],
    "O-Level Science": [o_level_pass_fail(o_level_science)],
    "Grade 1": [grade_to_points[a_level_grades["Grade 1"]]],
    "Grade 2": [grade_to_points[a_level_grades["Grade 2"]]],
    "Grade 3": [grade_to_points[a_level_grades["Grade 3"]]],
    "Program 1 Choice": [program_choices[0]],
    "Program 2 Choice": [program_choices[1]],
    "Program 3 Choice": [program_choices[2]],
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# One-hot encode categorical features
categorical_columns = ["Gender", "Program 1 Choice", "Program 2 Choice", "Program 3 Choice"]
input_encoded = pd.get_dummies(input_df, columns=categorical_columns)

# Align columns with training data
all_columns = pickle.load(open("columns.pkl", "rb"))  # Load columns from training phase
input_encoded = input_encoded.reindex(columns=all_columns, fill_value=0)

# Standardize numerical features
numerical_features = ["Grade 1", "Grade 2", "Grade 3"]
input_encoded[numerical_features] = scaler.transform(input_encoded[numerical_features])

# Prediction
if st.button("Predict Admission and Program"):
    # Admission Status Prediction
    admission_prediction = admission_model.predict(input_encoded)[0]
    admission_status = "Yes" if admission_prediction == 1 else "No"

    if admission_status == "Yes":
        # Program Prediction
        program_encoded = program_model.predict(input_encoded)[0]
        admitted_program = label_encoder.inverse_transform([program_encoded])[0]
        st.success(f"Admission Status: {admission_status}")
        st.success(f"Admitted Program: {admitted_program}")
    else:
        st.warning("Admission Status: No")
        st.warning("Not Admitted to any Program")
