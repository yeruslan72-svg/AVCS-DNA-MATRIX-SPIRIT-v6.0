# ============================================================
# 🧠 AVCS DNA-MATRIX SPIRIT — Unified UI Dashboard (v7.x)
# ============================================================
# Combines: Twin Control · Analytics · Adaptive Learning Layer
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# ======== Core Imports ========
from digital_twin.industrial_digital_twin import DigitalTwin
from industrial_core.data_manager import DataManager

# ======== Adaptive Learning Layer ========
from adaptive_learning.adaptive_engine import analyze_learning_progress
from adaptive_learning.feedback_controller import update_feedback_score
from adaptive_learning.sample_data import get_recent_activity

# ============================================================
# ⚙️ Initialization
# ============================================================

st.set_page_config(
    page_title="AVCS DNA-MATRIX SPIRIT Dashboard",
    page_icon="🧬",
    layout="wide",
)

# Sidebar
st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.title("AVCS DNA-MATRIX SPIRIT")
st.sidebar.markdown("### _Operational Excellence Delivered._")
st.sidebar.divider()
section = st.sidebar.radio(
    "Navigate:",
    ["🏭 Twin Control", "📊 Analytics", "🧬 Adaptive Learning"],
)

# ============================================================
# 🏭 DIGITAL TWIN PANEL
# ============================================================
if section == "🏭 Twin Control":
    st.title("🏭 Digital Twin — Real-Time Control Panel")

    twin = DigitalTwin()
    st.success("Digital Twin system synchronized successfully.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sensor Data Simulation")
        temp = st.slider("Temperature (°C)", 0, 120, 65)
        pressure = st.slider("Pressure (bar)", 0, 50, 20)
        flow = st.slider("Flow Rate (m³/h)", 0, 300, 150)

        status = twin.update_state(temp=temp, pressure=pressure, flow=flow)
        st.metric("System Status", status)

    with col2:
        st.subheader("Twin Visualization")
        chart_data = pd.DataFrame({
            "Temperature": [temp + np.random.randn()],
            "Pressure": [pressure + np.random.randn()],
            "Flow": [flow + np.random.randn()],
        })
        st.line_chart(chart_data)

    if st.button("🔄 Sync with Live System"):
        st.info("Synchronizing with PLC network...")
        time.sleep(1.5)
        st.success("Synchronization complete ✅")

# ============================================================
# 📊 ANALYTICS PANEL
# ============================================================
elif section == "📊 Analytics":
    st.title("📊 Operational Analytics Dashboard")

    data_path = "data/analytics_log.csv"
    os.makedirs("data", exist_ok=True)

    # Create sample data if missing
    if not os.path.exists(data_path) or os.stat(data_path).st_size == 0:
        st.info("Initializing analytics dataset...")
        timestamps = pd.date_range(end=datetime.now(), periods=50, freq="T")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "temp": np.random.uniform(45, 80, len(timestamps)),
            "pressure": np.random.uniform(15, 30, len(timestamps)),
            "flow": np.random.uniform(120, 260, len(timestamps)),
        })
        df.to_csv(data_path, index=False)
        st.success("Demo analytics data generated successfully.")
    else:
        df = pd.read_csv(data_path)

    if not df.empty:
        st.subheader("Performance Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Temperature", f"{df['temp'].mean():.2f} °C")
        col2.metric("Avg Pressure", f"{df['pressure'].mean():.2f} bar")
        col3.metric("Avg Flow", f"{df['flow'].mean():.2f} m³/h")

        st.line_chart(df.set_index("timestamp")[["temp", "pressure", "flow"]])
        st.caption("Live data visualization from analytics log.")
    else:
        st.warning("No data available — waiting for stream input...")

    if st.button("🧹 Clear Analytics Log"):
        open(data_path, "w").close()
        st.success("Analytics log cleared successfully.")

# ============================================================
# 🧬 ADAPTIVE LEARNING PANEL (SPIRIT v7.x)
# ============================================================
elif section == "🧬 Adaptive Learning":
    st.title("🧬 Adaptive Learning Layer — SPIRIT v7.x")
    st.markdown(
        "> The self-evolving intelligence that learns from feedback, context, and operational data."
    )

    with st.expander("📈 Adaptive Intelligence Dashboard", expanded=True):
        activity_data = get_recent_activity()
        feedback_score = update_feedback_score(activity_data)
        learning_status = analyze_learning_progress(activity_data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Feedback Score", f"{feedback_score:.2f}")
        col2.metric("Learning Progress", f"{learning_status['progress']}%")
        col3.metric("Pattern Recognition", learning_status['pattern'])

        st.line_chart(learning_status["trend"])

        if st.button("🔁 Retrain Adaptive Model"):
            with st.spinner("Retraining adaptive intelligence..."):
                time.sleep(2)
            st.success("SPIRIT Intelligence successfully updated! 🤖")

    st.caption("SPIRIT Layer actively evolves based on system interaction and feedback loops.")

# ============================================================
# 🪐 FOOTER
# ============================================================
st.divider()
st.markdown(
    "<center>© 2025 AVCS Systems | DNA-MATRIX SPIRIT v7.x</center>",
    unsafe_allow_html=True,
)
