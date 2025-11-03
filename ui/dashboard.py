import streamlit as st
from PIL import Image

# ================== CONFIG ==================
st.set_page_config(
    page_title="AVCS DNA-MATRIX SPIRIT",
    page_icon="🧠",
    layout="wide"
)

# ================== HEADER ==================
col1, col2 = st.columns([1, 6])
with col1:
    logo = Image.open("assets/logo.png")
    st.image(logo, width=80)
with col2:
    st.markdown("## **AVCS DNA-MATRIX SPIRIT**")
    st.markdown("*Operational Excellence Delivered...*")

st.divider()

# ================== SIDEBAR ==================
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Select Section:",
    ["Overview", "Digital Twin", "PLC Integration", "AI Core", "Logs"]
)

# ================== CONTENT ==================
if menu == "Overview":
    st.subheader("System Overview")
    st.info("Welcome to AVCS DNA-MATRIX SPIRIT v6.0 dashboard.")
elif menu == "Digital Twin":
    st.subheader("Digital Twin Module")
    st.write("Monitoring virtual plant models and live data streams...")
elif menu == "PLC Integration":
    st.subheader("PLC Integration Layer")
    st.write("Adapters and live PLC communication states...")
elif menu == "AI Core":
    st.subheader("AI Data Matrix Core")
    st.write("Processing, analytics, and model orchestration.")
elif menu == "Logs":
    st.subheader("System Logs")
    st.write("Event and status logs will appear here.")
