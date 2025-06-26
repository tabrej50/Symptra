import streamlit as st
from symptra_engine import symptra_chat

st.set_page_config(page_title="🩺 Symptra – AI Doctor", layout="wide")

st.title("🩺 Symptra – AI Clinical Assistant")

with st.form("symptom_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area("🩺 Symptoms", placeholder="e.g., Patient presents with chest pain, shortness of breath...")

    with col2:
        history = st.text_area("📜 Medical History", placeholder="e.g., History of hypertension, diabetes, MI in 2020...")
        medications = st.text_area("💊 Current Medications (one per line)", placeholder="e.g.,\nMetformin 500mg\nAspirin 81mg")

    submitted = st.form_submit_button("🧠 Get AI Opinion")

if submitted:
    with st.spinner("Symptra is analyzing..."):
        full_prompt = f"""A {age}-year-old {gender.lower()} presents with the following:

Symptoms:
{symptoms}

Medical History:
{history}

Current Medications:
{medications}

What is your structured clinical assessment?
"""
        result = symptra_chat(full_prompt)
        st.markdown("### 🩺 Symptra's Assessment")
        st.write(result)
