
import streamlit as st
from symptra_engine import symptra_chat

st.set_page_config(page_title="Symptra – AI Doctor", page_icon="🩺", layout="wide")
st.title("🩺 Symptra – AI Doctor")

query = st.text_input("Ask your medical question:")
if query:
    with st.spinner("Analyzing..."):
        answer = symptra_chat(query)
    st.markdown(answer)
