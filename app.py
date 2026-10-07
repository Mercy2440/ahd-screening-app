import streamlit as st
import pandas as pd
import joblib

# Set Page Configuration
st.set_page_config(
    page_title="AHD Clinical Decision Support Tool",
    page_icon="🩺",
    layout="centered"
)
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 Restricted Access")
        st.write("Please enter the password to access the AHD Screener.")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if password == "2440": 
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password")
        return False
    return True
if not check_password():
    st.stop()
# Load the trained ML Pipeline
@st.cache_resource
def load_model():
    return joblib.load('ahd_screening_model.pkl')

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading the model file. Please ensure 'ahd_screening_model.pkl' is in the workspace. {e}")
    st.stop()

# Header UI
st.title("🩺 Advanced HIV Disease (AHD) Screener")
st.markdown("""
This decision support tool uses a validated machine learning pipeline to screen and predict the risk of
**Advanced HIV Disease (AHD)** in adult clients, helping clinical teams prioritize patients for immediate confirmatory testing.
""")

st.divider()

# User Inputs
st.subheader("🧑‍⚕️ Enter Patient Clinical Metrics")
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sex", options=["Female", "Male"], index=0)
    age = st.number_input("Age at Reporting", min_value=1, max_value=120, value=35, step=1)
    current_regimen = st.selectbox("Current Regimen Line", options=["First line", "Second line", "Third line"], index=0)

with col2:
    baseline_cd4 = st.number_input("Baseline CD4 Count (cells/mm³)", min_value=0.0, max_value=3000.0, value=150.0, step=10.0)
    weight = st.number_input("Weight (kg)", min_value=5.0, max_value=200.0, value=60.0, step=0.5)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0, step=1.0)

# Dynamically calculate BMI to match training inputs
bmi = weight / ((height / 100) ** 2)
st.info(f"**Calculated Patient BMI:** {bmi:.2f} kg/m²")

# Predict Button
if st.button("Run Clinical Evaluation", type="primary"):
    # Transform inputs to match training dataframe formatting
    patient_data = {
        'Age at Reporting': [age],
        'Baseline CD4 result': [baseline_cd4],
        'BMI': [bmi],
        'Sex_M': [True if sex == "Male" else False],
        'Current Regimen Line_Second line': [True if current_regimen == "Second line" else False],
        'Current Regimen Line_Third line': [True if current_regimen == "Third line" else False]
    }

    patient_df = pd.DataFrame(patient_data)

    # Predict Probability and classification status
    probability = pipeline.predict_proba(patient_df)[0, 1]

    st.divider()
    st.subheader("📊 Diagnostic Assessment Results")

    # Display Metric Card
    st.metric(label="AHD Risk Probability", value=f"{probability:.1%}")

    # Dynamic output based on 40% clinical threshold
    if probability >= 0.40:
        st.error("⚠️ **Result: HIGH RISK OF AHD**")
        st.markdown("""
        * **Clinical Recommendation:** Flag this client immediately. Prioritize for an immediate AHD clinical package and confirmatory laboratory testing.
        """)
    else:
        st.success("✅ **Result: Standard Risk**")
        st.markdown("""
        * **Clinical Recommendation:** Maintain standard clinical care pathways and routine periodic diagnostic monitoring.
        """)
