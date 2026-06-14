import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# Header
st.title("💸 Employee Salary Prediction")
st.markdown(
    "Predict employee salaries based on their experience, education, role, and industry."
)

st.write("✅ Step 1: App started")

@st.cache_data
def load_data_choices():
    st.write("📂 Reading dataset...")

    df = pd.read_csv("job_salary_prediction_dataset.csv")

    choices = {
        "job_title": sorted(df["job_title"].dropna().unique().tolist()),
        "education_level": sorted(df["education_level"].dropna().unique().tolist()),
        "industry": sorted(df["industry"].dropna().unique().tolist()),
        "company_size": sorted(df["company_size"].dropna().unique().tolist()),
        "location": sorted(df["location"].dropna().unique().tolist()),
        "remote_work": sorted(df["remote_work"].dropna().unique().tolist())
    }

    return choices

@st.cache_resource
def load_model():
    st.write("🤖 Starting model load...")

    model = joblib.load("model.pkl")

    st.write("✅ Model loaded successfully!")

    return model

try:
    st.write("⏳ Loading dropdown data...")
    choices = load_data_choices()
    st.write("✅ Step 2: Dropdown data loaded")

    st.write("⏳ Loading model...")
    model = load_model()
    st.write("✅ Step 3: Model loaded")

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Professional Details")

    job_title = st.selectbox(
        "Job Title",
        choices["job_title"]
    )

    experience_years = st.slider(
        "Experience (Years)",
        min_value=0,
        max_value=50,
        value=5
    )

    education_level = st.selectbox(
        "Education Level",
        choices["education_level"]
    )

    skills_count = st.slider(
        "Number of Skills",
        min_value=0,
        max_value=50,
        value=5
    )

    certifications = st.slider(
        "Number of Certifications",
        min_value=0,
        max_value=20,
        value=1
    )

with col2:
    st.subheader("Company & Location")

    industry = st.selectbox(
        "Industry",
        choices["industry"]
    )

    company_size = st.selectbox(
        "Company Size",
        choices["company_size"]
    )

    location = st.selectbox(
        "Location",
        choices["location"]
    )

    remote_work = st.selectbox(
        "Remote Work",
        choices["remote_work"]
    )

st.markdown("---")

if st.button("Predict Salary 🚀", type="primary"):

    input_data = pd.DataFrame({
        "job_title": [job_title],
        "experience_years": [experience_years],
        "education_level": [education_level],
        "skills_count": [skills_count],
        "industry": [industry],
        "company_size": [company_size],
        "location": [location],
        "remote_work": [remote_work],
        "certifications": [certifications]
    })

    with st.spinner("Calculating..."):
        prediction = model.predict(input_data)[0]

    st.success(f"### Estimated Salary: ₹{prediction:,.2f}")
    st.balloons()