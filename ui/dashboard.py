import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# ===============================
# AVCS DNA-MATRIX SPIRIT v6.0 UI
# ===============================

st.set_page_config(
    page_title="AVCS DNA-MATRIX SPIRIT",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# ---------- ASSETS ----------
logo_path = Path("assets/logo.png")
if logo_path.exists():
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_column_width=True)
st.sidebar.markdown("### **AVCS DNA-MATRIX SPIRIT v6.0**")
st.sidebar.caption("Operational Excellence Delivered...")

# ---------- DATA HANDLING ----------
analytics_file = Path("data/analytics_log.csv")
os.makedirs("data", exist_ok=True)
if not analytics_file.exists():
    with open(analytics_file, "w") as f:
        f.write("timestamp,health_score,risk_index,rul_hours,anomaly,recommended_action\n")

# ---------- MAIN SECTIONS ----------
tabs = st.tabs(["🏭 Digital Twin Control", "📊 Analytics Panel", "⚡ System Health"])

# ---------- DIGITAL TWIN CONTROL ----------
with tabs[0]:
    st.header("🏭 Digital Twin Control")
    st.write("Monitor and interact with real-time virtualized industrial assets.")

    col1, col2, col3 = st.columns(3)
    with col1:
        pressure = st.slider("Pressure [bar]", 0.0, 10.0, 4.5, step=0.1)
    with col2:
        temp = st.slider("Temperature [°C]", 20.0, 120.0, 65.0, step=0.5)
    with col3:
        vibration = st.slider("Vibration [mm/s]", 0.0, 50.0, 7.5, step=0.5)

    # Calculate Health Metrics
    health_score = max(0, 100 - (vibration * 0.7 + (temp - 60) * 0.5 + (pressure - 5) * 1.5))
    risk_index = np.clip((100 - health_score) / 10, 0, 10)
    rul_hours = np.clip((health_score / 100) * 5000, 0, 5000)
    anomaly = "⚠️ High vibration" if vibration > 40 else "✅ Stable"

    # Save analytics
    new_row = f"{datetime.now().isoformat(timespec='seconds')},{health_score:.2f},{risk_index:.2f},{rul_hours:.0f},{anomaly},None\n"
    with open(analytics_file, "a") as f:
        f.write(new_row)

    # Display metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Health Score", f"{health_score:.1f} %")
    c2.metric("Risk Index", f"{risk_index:.1f} / 10")
    c3.metric("Estimated RUL", f"{rul_hours:.0f} h")

    # Dynamic Gauge
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'indicator'}]])
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "System Health"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "mediumseagreen"},
               'steps': [
                   {'range': [0, 40], 'color': "crimson"},
                   {'range': [40, 70], 'color': "gold"},
                   {'range': [70, 100], 'color': "lightgreen"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------- ANALYTICS PANEL ----------
with tabs[1]:
    st.header("📊 Analytics & Diagnostics")

    df = pd.read_csv(analytics_file)
    if len(df) > 0:
        st.dataframe(df.tail(10), use_container_width=True)

        # Trend chart
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["health_score"], name="Health Score", mode="lines+markers"))
        trend_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["risk_index"], name="Risk Index", mode="lines+markers", yaxis="y2"))

        trend_fig.update_layout(
            yaxis=dict(title="Health Score", range=[0, 100]),
            yaxis2=dict(title="Risk Index", overlaying="y", side="right", range=[0, 10]),
            legend=dict(x=0.5, y=1.1, orientation="h"),
            margin=dict(l=40, r=40, t=50, b=40),
            template="plotly_white"
        )
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("No analytics data recorded yet.")

# ---------- SYSTEM HEALTH ----------
with tabs[2]:
    st.header("⚡ System Health Overview")

    st.markdown("""
    **System Components:**
    - Digital Twin Engine – ✅ Running  
    - PLC Integration Layer – ✅ Connected  
    - Data Manager – ✅ Active  
    - AI Predictive Module – 🔄 Training in background  

    **Next Maintenance Window:** 350 h  
    **Firmware Version:** v6.0.14  
    """)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("© 2025 AVCS DNA-MATRIX SPIRIT | Industrial Intelligence Redefined")

