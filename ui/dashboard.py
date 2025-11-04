"""
dashboard.py
AVCS DNA-MATRIX SPIRIT v6.1a — SPIRIT FRAME
Streamlit dashboard with integrated Digital Twin Control and Analytics Engine.
Now includes automatic CSV logging for analytics history.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import random
from datetime import datetime

from industrial_core.data_manager import DataBuffer
from industrial_core.analytics_engine import AnalyticsEngine
from ui.twin_control_panel import render_twin_control

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AVCS DNA-MATRIX SPIRIT v6.1a",
    layout="wide",
    page_icon="🧠",
)

DATA_PATH = "data/analytics_log.csv"
os.makedirs("data", exist_ok=True)

# =====================================================
# INITIALIZATION
# =====================================================

if "sensors_buf" not in st.session_state:
    st.session_state["sensors_buf"] = DataBuffer(maxlen=500)

if "streaming" not in st.session_state:
    st.session_state["streaming"] = False

if "analytics" not in st.session_state:
    st.session_state["analytics"] = AnalyticsEngine()

if "last_analytics" not in st.session_state:
    st.session_state["last_analytics"] = None

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.markdown("### AVCS DNA-MATRIX SPIRIT FRAME")
st.sidebar.markdown("**Operational Excellence Delivered** 🚀")

menu = st.sidebar.radio(
    "Select module:",
    ["Dashboard", "Digital Twin", "Analytics", "History", "Settings"],
)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "Dashboard":
    st.title("📊 AVCS DNA-MATRIX SPIRIT Dashboard")
    st.markdown("### Real-time equipment condition monitoring")

    col1, col2, col3 = st.columns(3)
    buf = st.session_state["sensors_buf"]

    with col1:
        st.metric("Temperature (°C)", f"{buf.temperature[-1] if buf.temperature else 0:.2f}")
    with col2:
        st.metric("Pressure (bar)", f"{buf.pressure[-1] if buf.pressure else 0:.2f}")
    with col3:
        st.metric("Vibration (g)", f"{buf.vibration[-1] if buf.vibration else 0:.3f}")

    st.markdown("---")
    chart_placeholder = st.empty()

    start_btn, stop_btn = st.columns(2)
    with start_btn:
        if st.button("▶️ Start Sensors Stream", type="primary"):
            st.session_state["streaming"] = True
    with stop_btn:
        if st.button("⏹ Stop Stream"):
            st.session_state["streaming"] = False

    if st.session_state["streaming"]:
        st.info("Streaming sensor data... press Stop to pause.")
        chart_data = pd.DataFrame(columns=["timestamp", "temperature", "pressure", "vibration"])

        for _ in range(100):
            if not st.session_state["streaming"]:
                break
            ts = datetime.utcnow()
            temp = 60 + random.random() * 20
            pres = 4 + random.random() * 2
            vib = 0.2 + random.random() * 2
            st.session_state["sensors_buf"].append(ts, temp, pres, vib)
            chart_data = chart_data._append(
                {"timestamp": ts, "temperature": temp, "pressure": pres, "vibration": vib},
                ignore_index=True
            )
            chart_placeholder.line_chart(
                chart_data.set_index("timestamp")[["temperature", "pressure", "vibration"]]
            )
            time.sleep(0.3)
        st.success("Stream stopped.")
    else:
        if len(buf.timestamps) > 0:
            df = pd.DataFrame({
                "timestamp": buf.timestamps,
                "temperature": buf.temperature,
                "pressure": buf.pressure,
                "vibration": buf.vibration
            })
            st.line_chart(df.set_index("timestamp"))
        else:
            st.info("No sensor data yet — start the stream to begin monitoring.")

# =====================================================
# DIGITAL TWIN
# =====================================================

elif menu == "Digital Twin":
    st.title("🧩 Digital Twin Control Center")
    st.markdown("Configure and simulate operational scenarios.")
    render_twin_control()

# =====================================================
# ANALYTICS PANEL
# =====================================================

elif menu == "Analytics":
    st.title("🧠 Predictive Analytics & Health Assessment")
    buf = st.session_state["sensors_buf"]
    engine = st.session_state["analytics"]

    if len(buf.timestamps) > 0:
        latest = {
            "vibration": buf.vibration[-1],
            "temperature": buf.temperature[-1],
            "pressure": buf.pressure[-1],
        }
        result = engine.predict(latest)
        st.session_state["last_analytics"] = result

        # ✅ Log result to CSV
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "health_score": result["health_score"],
            "risk_index": result["risk_index"],
            "rul_hours": result["rul_hours"],
            "anomaly": result["anomaly"],
            "recommended_action": result["recommended_action"],
        }

        df_new = pd.DataFrame([record])
        if os.path.exists(DATA_PATH):
            df_old = pd.read_csv(DATA_PATH)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(DATA_PATH, index=False)

        st.success("✅ Analytics saved to data/analytics_log.csv")

        # Display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Health Score", f"{result['health_score']}%")
        col2.metric("Risk Index", f"{result['risk_index']}%")
        col3.metric("RUL Estimate", f"{result['rul_hours']} h")
        col4.metric("Anomaly", "⚠️ Yes" if result["anomaly"] else "✅ No")

        st.markdown("### Detailed Analysis")
        st.json(result)
    else:
        st.info("No data — start streaming or run Twin simulation first.")

# =====================================================
# HISTORY VIEW
# =====================================================

elif menu == "History":
    st.title("📚 Analytics History Log")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.dataframe(df.tail(50))
        st.download_button(
            "📥 Download Full CSV",
            data=open(DATA_PATH, "rb"),
            file_name="analytics_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No analytics history found yet.")

# =====================================================
# SETTINGS
# =====================================================

elif menu == "Settings":
    st.title("⚙️ System Settings & Info")
    st.markdown("### AVCS DNA-MATRIX SPIRIT FRAME v6.1a")
    st.markdown(
        """
        **Core Modules:**
        - `industrial_core` — Data Manager, Analytics Engine  
        - `digital_twin` — Equipment Behavior Simulation  
        - `plc_integration` — OPC/Modbus Connectivity  
        - `ui` — Streamlit Frontend Components  
        - `assets/logo.png` — Branding and identity  
        - `data/analytics_log.csv` — Analytics result history  

        **Operational Excellence Delivered.**
        """
    )
    st.caption(f"Build timestamp: {datetime.utcnow().isoformat()}Z")
