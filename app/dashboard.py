import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os

# --------------------------
# Load model
# --------------------------
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "latest_model.pkl")
model = joblib.load(model_path)

# --------------------------
# Load dataset from SQL
# --------------------------
db_path = os.path.join(os.path.dirname(__file__), "..", "database", "gym_data.db")
conn = sqlite3.connect(db_path)
df = pd.read_sql("SELECT * FROM gym_willingness", conn)
conn.close()

# --------------------------
# Page Title
# --------------------------
st.title("Gym Willingness Prediction System")

st.write(
"This system predicts whether a student is willing to go to the gym "
"based on lifestyle and behavioral factors."
)

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("Enter Student Details")

study_hours = st.sidebar.slider("Study Hours",1,8,4)
exam_soon = st.sidebar.selectbox("Exam Soon?", [0,1])
sleep_hours = st.sidebar.slider("Sleep Hours",4,9,7)
stress_level = st.sidebar.slider("Stress Level",1,10,5)
energy_level = st.sidebar.slider("Energy Level",1,10,5)
past_week_gym = st.sidebar.slider("Past Week Gym Visits",0,7,3)

# --------------------------
# Prediction Button
# --------------------------
if st.sidebar.button("Predict"):

    input_data = pd.DataFrame({
        "STUDY_HOURS":[study_hours],
        "EXAM_SOON":[exam_soon],
        "SLEEP_HOURS":[sleep_hours],
        "STRESS_LEVEL":[stress_level],
        "ENERGY_LEVEL":[energy_level],
        "PAST_WEEK_GYM_VISITS":[past_week_gym]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("Student is willing to go to the gym")
    else:
        st.error("Student is NOT willing to go to the gym")

    st.write("Probability:", round(probability,2))

# --------------------------
# Dataset Preview
# --------------------------
st.subheader("Dataset Preview")

st.dataframe(df.head())

# --------------------------
# Dataset Statistics
# --------------------------
st.subheader("Dataset Statistics")

st.write(df.describe())